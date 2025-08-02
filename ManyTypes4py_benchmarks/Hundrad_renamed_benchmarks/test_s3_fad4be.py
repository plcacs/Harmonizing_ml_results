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
aws_clients = ['aws_client_parameters_custom_endpoint',
    'aws_client_parameters_empty', 'aws_client_parameters_public_bucket']


@pytest.fixture
def func_w8y2pf1j(monkeypatch, client_parameters):
    if client_parameters.endpoint_url:
        monkeypatch.setenv('MOTO_S3_CUSTOM_ENDPOINTS', client_parameters.
            endpoint_url)
    with mock_s3():
        yield


@pytest.fixture
def func_oqydgy52(request):
    client_parameters = request.getfixturevalue(request.param)
    return client_parameters


@pytest.fixture
def func_5nqqhigu(s3_mock, request):
    s3 = boto3.resource('s3')
    bucket = s3.Bucket('bucket')
    marker = request.node.get_closest_marker('is_public', None)
    if marker and marker.args[0]:
        func_5nqqhigu.create(ACL='public-read')
    else:
        func_5nqqhigu.create()
    return bucket


@pytest.fixture
def func_9opkvcbd(s3_mock, request):
    s3 = boto3.resource('s3')
    bucket = s3.Bucket('bucket_2')
    marker = request.node.get_closest_marker('is_public', None)
    if marker and marker.args[0]:
        func_5nqqhigu.create(ACL='public-read')
    else:
        func_5nqqhigu.create()
    return bucket


@pytest.fixture
def func_4y3gnlxv(bucket, tmp_path):
    file = tmp_path / 'object.txt'
    file.write_text('TEST')
    with open(file, 'rb') as f:
        return func_5nqqhigu.upload_fileobj(f, 'object')


@pytest.fixture
def func_5isiel7m(bucket, tmp_path):
    file = tmp_path / 'object_in_folder.txt'
    file.write_text('TEST OBJECT IN FOLDER')
    with open(file, 'rb') as f:
        return func_5nqqhigu.upload_fileobj(f, 'folder/object')


@pytest.fixture
def func_6dlmb7cl(bucket, tmp_path):
    objects = []
    for filename in ['folderobject/foo.txt', 'folderobject/bar.txt',
        'folder/object/foo.txt', 'folder/object/bar.txt']:
        file = tmp_path / filename
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text('TEST OBJECTS IN FOLDER')
        with open(file, 'rb') as f:
            filename = Path(filename)
            obj = func_5nqqhigu.upload_fileobj(f, (filename.parent /
                filename.stem).as_posix())
            objects.append(obj)
    return objects


@pytest.fixture
def func_fgsqzroa(bucket, tmp_path):
    objects = []
    for i in range(0, 20):
        file = tmp_path / f'object{i}.txt'
        file.write_text('TEST')
        with open(file, 'rb') as f:
            objects.append(func_5nqqhigu.upload_fileobj(f, f'object{i}'))
    return objects


@pytest.mark.parametrize('client_parameters', [
    'aws_client_parameters_custom_endpoint'], indirect=True)
async def func_khgufs5p(object, client_parameters, aws_credentials):
    client_parameters_wrong_endpoint = AwsClientParameters(endpoint_url=
        'http://something')

    @flow
    async def func_fo3hlkj0():
        return await s3_download(bucket='bucket', key='object',
            aws_credentials=aws_credentials, aws_client_parameters=
            client_parameters_wrong_endpoint)
    with pytest.raises(EndpointConnectionError):
        await func_fo3hlkj0()


@pytest.mark.parametrize('client_parameters', [pytest.param(
    'aws_client_parameters_custom_endpoint', marks=pytest.mark.is_public(
    False)), pytest.param('aws_client_parameters_custom_endpoint', marks=
    pytest.mark.is_public(True)), pytest.param(
    'aws_client_parameters_empty', marks=pytest.mark.is_public(False)),
    pytest.param('aws_client_parameters_empty', marks=pytest.mark.is_public
    (True)), pytest.param('aws_client_parameters_public_bucket', marks=[
    pytest.mark.is_public(False), pytest.mark.xfail(reason=
    'Bucket is not a public one')]), pytest.param(
    'aws_client_parameters_public_bucket', marks=pytest.mark.is_public(True
    ))], indirect=True)
async def func_t4tc0d72(object, client_parameters, aws_credentials):

    @flow
    async def func_fo3hlkj0():
        return await s3_download(bucket='bucket', key='object',
            aws_credentials=aws_credentials, aws_client_parameters=
            client_parameters)
    result = await func_fo3hlkj0()
    assert result == b'TEST'


@pytest.mark.parametrize('client_parameters', aws_clients, indirect=True)
async def func_u0hwpofj(object, client_parameters, aws_credentials):

    @flow
    async def func_fo3hlkj0():
        return await s3_download(key='unknown_object', bucket='bucket',
            aws_credentials=aws_credentials, aws_client_parameters=
            client_parameters)
    with pytest.raises(ClientError):
        await func_fo3hlkj0()


@pytest.mark.parametrize('client_parameters', aws_clients, indirect=True)
async def func_e423p9vi(bucket, client_parameters, tmp_path, aws_credentials):

    @flow
    async def func_fo3hlkj0():
        test_file = tmp_path / 'test.txt'
        test_file.write_text('NEW OBJECT')
        with open(test_file, 'rb') as f:
            return await s3_upload(data=f.read(), bucket='bucket', key=
                'new_object', aws_credentials=aws_credentials,
                aws_client_parameters=client_parameters)
    await func_fo3hlkj0()
    stream = io.BytesIO()
    func_5nqqhigu.download_fileobj('new_object', stream)
    stream.seek(0)
    output = stream.read()
    assert output == b'NEW OBJECT'


@pytest.mark.parametrize('client_parameters', aws_clients, indirect=True)
async def func_qucdb42m(object, bucket, bucket_2, aws_credentials):

    def func_fqsspnl3(bucket, key):
        stream = io.BytesIO()
        func_5nqqhigu.download_fileobj(key, stream)
        stream.seek(0)
        return stream.read()

    @flow
    async def func_fo3hlkj0():
        await s3_copy(source_path='object', target_path=
            'subfolder/new_object', source_bucket_name='bucket',
            aws_credentials=aws_credentials, target_bucket_name='bucket_2')
        await s3_copy(source_path='object', target_path=
            'subfolder/new_object', source_bucket_name='bucket',
            aws_credentials=aws_credentials)
    await func_fo3hlkj0()
    assert func_fqsspnl3(bucket_2, 'subfolder/new_object') == b'TEST'
    assert func_fqsspnl3(bucket, 'subfolder/new_object') == b'TEST'


@pytest.mark.parametrize('client_parameters', aws_clients, indirect=True)
async def func_ykxlieqt(object, bucket, bucket_2, aws_credentials):

    def func_fqsspnl3(bucket, key):
        stream = io.BytesIO()
        func_5nqqhigu.download_fileobj(key, stream)
        stream.seek(0)
        return stream.read()

    @flow
    async def func_fo3hlkj0():
        await s3_move(source_path='object', target_path=
            'subfolder/object_copy', source_bucket_name='bucket',
            aws_credentials=aws_credentials)
        await s3_move(source_path='subfolder/object_copy', target_path=
            'object_copy_2', source_bucket_name='bucket',
            target_bucket_name='bucket_2', aws_credentials=aws_credentials)
    await func_fo3hlkj0()
    assert func_fqsspnl3(bucket_2, 'object_copy_2') == b'TEST'
    with pytest.raises(ClientError):
        func_fqsspnl3(bucket, 'object')
    with pytest.raises(ClientError):
        func_fqsspnl3(bucket, 'subfolder/object_copy')


@pytest.mark.parametrize('client_parameters', aws_clients, indirect=True)
async def func_gunic4tc(object, bucket, aws_credentials):

    def func_fqsspnl3(bucket, key):
        stream = io.BytesIO()
        func_5nqqhigu.download_fileobj(key, stream)
        stream.seek(0)
        return stream.read()

    @flow
    async def func_fo3hlkj0():
        await s3_move(source_path='object', target_path=
            'subfolder/new_object', source_bucket_name='bucket',
            aws_credentials=aws_credentials, target_bucket_name=
            'nonexistent-bucket')
    with pytest.raises(ClientError):
        await func_fo3hlkj0()
    assert func_fqsspnl3(bucket, 'object') == b'TEST'


@pytest.mark.parametrize('client_parameters', aws_clients, indirect=True)
async def func_0lmbgokw(object, bucket, aws_credentials):

    def func_fqsspnl3(bucket, key):
        stream = io.BytesIO()
        func_5nqqhigu.download_fileobj(key, stream)
        stream.seek(0)
        return stream.read()

    @flow
    async def func_fo3hlkj0(source_path, target_path, source_bucket_name,
        target_bucket_name):
        await s3_move(source_path=source_path, target_path=target_path,
            source_bucket_name=source_bucket_name, aws_credentials=
            aws_credentials, target_bucket_name=target_bucket_name)
    with pytest.raises(ClientError):
        await func_fo3hlkj0(source_path='object', target_path=
            'subfolder/new_object', source_bucket_name='bucket',
            target_bucket_name='nonexistent-bucket')
    assert func_fqsspnl3(bucket, 'object') == b'TEST'
    with pytest.raises(ClientError):
        await func_fo3hlkj0(source_path='object', target_path='object',
            source_bucket_name='bucket', target_bucket_name='bucket')
    assert func_fqsspnl3(bucket, 'object') == b'TEST'


@pytest.mark.parametrize('client_parameters', aws_clients, indirect=True)
async def func_0copefpe(object, client_parameters, object_in_folder,
    aws_credentials):

    @flow
    async def func_fo3hlkj0():
        return await s3_list_objects(bucket='bucket', aws_credentials=
            aws_credentials, aws_client_parameters=client_parameters)
    objects = await func_fo3hlkj0()
    assert len(objects) == 2
    assert [object['Key'] for object in objects] == ['folder/object', 'object']


@pytest.mark.parametrize('client_parameters', aws_clients, indirect=True)
async def func_wemisfg7(a_lot_of_objects, client_parameters, aws_credentials):

    @flow
    async def func_fo3hlkj0():
        return await s3_list_objects(bucket='bucket', aws_credentials=
            aws_credentials, aws_client_parameters=client_parameters,
            page_size=2)
    objects = await func_fo3hlkj0()
    assert len(objects) == 20
    assert sorted([object['Key'] for object in objects]) == sorted([
        f'object{i}' for i in range(0, 20)])


@pytest.mark.parametrize('client_parameters', aws_clients, indirect=True)
async def func_bcwd3c03(object, client_parameters, object_in_folder,
    aws_credentials):

    @flow
    async def func_fo3hlkj0():
        return await s3_list_objects(bucket='bucket', prefix='folder',
            aws_credentials=aws_credentials, aws_client_parameters=
            client_parameters)
    objects = await func_fo3hlkj0()
    assert len(objects) == 1
    assert [object['Key'] for object in objects] == ['folder/object']


@pytest.mark.parametrize('client_parameters', aws_clients, indirect=True)
async def func_qkw6m4im(object, client_parameters, objects_in_folder,
    aws_credentials):

    @flow
    async def func_fo3hlkj0(slash=False):
        return await s3_list_objects(bucket='bucket', prefix='folder' + (
            '/' if slash else ''), aws_credentials=aws_credentials,
            aws_client_parameters=client_parameters)
    objects = await func_fo3hlkj0(slash=True)
    assert len(objects) == 2
    assert [object['Key'] for object in objects] == ['folder/object/bar',
        'folder/object/foo']
    objects = await func_fo3hlkj0(slash=False)
    assert len(objects) == 4
    assert [object['Key'] for object in objects] == ['folder/object/bar',
        'folder/object/foo', 'folderobject/bar', 'folderobject/foo']


@pytest.mark.parametrize('client_parameters', aws_clients, indirect=True)
async def func_6uu11o3q(object, client_parameters, object_in_folder,
    aws_credentials):

    @flow
    async def func_fo3hlkj0():
        return await s3_list_objects(bucket='bucket', jmespath_query=
            'Contents[?Size > `10`][]', aws_credentials=aws_credentials,
            aws_client_parameters=client_parameters)
    objects = await func_fo3hlkj0()
    assert len(objects) == 1
    assert [object['Key'] for object in objects] == ['folder/object']


@pytest.fixture
def func_9adu11to():
    return AwsCredentials(aws_access_key_id='testing',
        aws_secret_access_key='testing')


@pytest.fixture
def func_mf6muq1k():
    return MinIOCredentials(minio_root_user='minioadmin',
        minio_root_password='minioadmin')


BUCKET_NAME = 'test_bucket'


@pytest.fixture
def func_z7tsis83():
    """Mock connection to AWS S3 with boto3 client."""
    with mock_s3():
        yield boto3.client(service_name='s3', region_name='us-east-1',
            aws_access_key_id='minioadmin', aws_secret_access_key='testing',
            aws_session_token='testing')


@pytest.fixture
def func_ax8i6vzw(s3, s3_bucket, tmp_path):
    """Creates an S3 bucket with multiple files in a nested structure"""
    file = tmp_path / 'object.txt'
    file.write_text('TEST')
    func_z7tsis83.upload_file(str(file), BUCKET_NAME, 'object.txt')
    func_z7tsis83.upload_file(str(file), BUCKET_NAME,
        'level1/object_level1.txt')
    func_z7tsis83.upload_file(str(file), BUCKET_NAME,
        'level1/level2/object_level2.txt')
    func_z7tsis83.upload_file(str(file), BUCKET_NAME,
        'level1/level2/object2_level2.txt')
    file.unlink()
    assert not file.exists()


@pytest.fixture(params=['aws_credentials', 'minio_credentials'])
def func_afklp0ml(s3, request, aws_creds_block, minio_creds_block):
    key = request.param
    if key == 'aws_credentials':
        fs = S3Bucket(bucket_name=BUCKET_NAME, credentials=aws_creds_block)
    elif key == 'minio_credentials':
        fs = S3Bucket(bucket_name=BUCKET_NAME, credentials=minio_creds_block)
    func_z7tsis83.create_bucket(Bucket=BUCKET_NAME)
    return fs


@pytest.fixture
def func_pyrb2u1t(s3_bucket):
    key = func_afklp0ml.write_path('test.txt', content=b'hello')
    return s3_bucket, key


async def func_atqcre54(s3_bucket):
    """
    Create an S3 bucket, instantiate S3Bucket block, write to and read from
    bucket.
    """
    key = await func_afklp0ml.write_path('test.txt', content=b'hello')
    assert await func_afklp0ml.read_path(key) == b'hello'


async def func_1qtyne92(s3_bucket):
    """
    Create an S3 bucket, instantiate S3Bucket block, write to path with
    missing directory.
    """
    key = await func_afklp0ml.write_path('folder/test.txt', content=b'hello')
    assert await func_afklp0ml.read_path(key) == b'hello'


async def func_amwekx11(s3_bucket):
    """
    Create an S3 bucket, instantiate S3Bucket block, assert read from
    nonexistent path fails.
    """
    with pytest.raises(ClientError):
        await func_afklp0ml.read_path('test_bucket/foo/bar')


@pytest.mark.parametrize('type_', [PureWindowsPath, PurePosixPath, str])
@pytest.mark.parametrize('delimiter', ['\\', '/'])
async def func_qkp837f9(s3_bucket, aws_creds_block, delimiter, type_):
    """Test the bucket folder functionality."""
    s3_bucket_block = S3Bucket(bucket_name=BUCKET_NAME, credentials=
        aws_creds_block, bucket_folder='subfolder/subsubfolder')
    key = await s3_bucket_block.write_path('test.txt', content=b'hello')
    assert await s3_bucket_block.read_path('test.txt') == b'hello'
    expected = 'subfolder/subsubfolder/test.txt'
    assert key == expected


async def func_x07jteor(nested_s3_bucket_structure, s3_bucket, tmp_path):
    await func_afklp0ml.get_directory(local_path=str(tmp_path))
    assert (tmp_path / 'object.txt').exists()
    assert (tmp_path / 'level1' / 'object_level1.txt').exists()
    assert (tmp_path / 'level1' / 'level2' / 'object_level2.txt').exists()
    assert (tmp_path / 'level1' / 'level2' / 'object2_level2.txt').exists()


async def func_2ggvlp4y(nested_s3_bucket_structure, s3_bucket, tmp_path,
    aws_creds_block):
    s3_bucket_block = S3Bucket(bucket_name=BUCKET_NAME, credentials=
        aws_creds_block, bucket_folder='level1/level2')
    await s3_bucket_block.get_directory(local_path=str(tmp_path))
    assert len(list(tmp_path.glob('*'))) == 2
    assert (tmp_path / 'object_level2.txt').exists()
    assert (tmp_path / 'object2_level2.txt').exists()


async def func_wdvmk3yf(nested_s3_bucket_structure, s3_bucket, tmp_path,
    aws_creds_block):
    await func_afklp0ml.get_directory(local_path=str(tmp_path), from_path=
        'level1')
    assert (tmp_path / 'object_level1.txt').exists()
    assert (tmp_path / 'level2' / 'object_level2.txt').exists()
    assert (tmp_path / 'level2' / 'object2_level2.txt').exists()


async def func_p6uzfvpl(s3_bucket, tmp_path):
    (tmp_path / 'file1.txt').write_text('FILE 1')
    (tmp_path / 'file2.txt').write_text('FILE 2')
    (tmp_path / 'folder1').mkdir()
    (tmp_path / 'folder1' / 'file3.txt').write_text('FILE 3')
    (tmp_path / 'folder1' / 'file4.txt').write_text('FILE 4')
    (tmp_path / 'folder1' / 'folder2').mkdir()
    (tmp_path / 'folder1' / 'folder2' / 'file5.txt').write_text('FILE 5')
    uploaded_file_count = await func_afklp0ml.put_directory(local_path=str(
        tmp_path))
    assert uploaded_file_count == 5
    (tmp_path / 'downloaded_files').mkdir()
    await func_afklp0ml.get_directory(local_path=str(tmp_path /
        'downloaded_files'))
    assert (tmp_path / 'downloaded_files' / 'file1.txt').exists()
    assert (tmp_path / 'downloaded_files' / 'file2.txt').exists()
    assert (tmp_path / 'downloaded_files' / 'folder1' / 'file3.txt').exists()
    assert (tmp_path / 'downloaded_files' / 'folder1' / 'file4.txt').exists()
    assert (tmp_path / 'downloaded_files' / 'folder1' / 'folder2' / 'file5.txt'
        ).exists()


async def func_8c4qcesi(s3_bucket, tmp_path, aws_creds_block):
    (tmp_path / 'file1.txt').write_text('FILE 1')
    (tmp_path / 'file2.txt').write_text('FILE 2')
    (tmp_path / 'folder1').mkdir()
    (tmp_path / 'folder1' / 'file3.txt').write_text('FILE 3')
    (tmp_path / 'folder1' / 'file4.txt').write_text('FILE 4')
    (tmp_path / 'folder1' / 'folder2').mkdir()
    (tmp_path / 'folder1' / 'folder2' / 'file5.txt').write_text('FILE 5')
    s3_bucket_block = S3Bucket(bucket_name=BUCKET_NAME, aws_credentials=
        aws_creds_block, basepath='subfolder')
    uploaded_file_count = await s3_bucket_block.put_directory(local_path=
        str(tmp_path))
    assert uploaded_file_count == 5
    (tmp_path / 'downloaded_files').mkdir()
    await s3_bucket_block.get_directory(local_path=str(tmp_path /
        'downloaded_files'))
    assert (tmp_path / 'downloaded_files' / 'file1.txt').exists()
    assert (tmp_path / 'downloaded_files' / 'file2.txt').exists()
    assert (tmp_path / 'downloaded_files' / 'folder1' / 'file3.txt').exists()
    assert (tmp_path / 'downloaded_files' / 'folder1' / 'file4.txt').exists()
    assert (tmp_path / 'downloaded_files' / 'folder1' / 'folder2' / 'file5.txt'
        ).exists()


async def func_c98piuhy(s3_bucket, tmp_path, aws_creds_block):
    (tmp_path / 'file1.txt').write_text('FILE 1')
    (tmp_path / 'file2.txt').write_text('FILE 2')
    (tmp_path / 'folder1').mkdir()
    (tmp_path / 'folder1' / 'file3.txt').write_text('FILE 3')
    (tmp_path / 'folder1' / 'file4.txt').write_text('FILE 4')
    (tmp_path / 'folder1' / 'folder2').mkdir()
    (tmp_path / 'folder1' / 'folder2' / 'file5.txt').write_text('FILE 5')
    (tmp_path / '.prefectignore').write_text('folder2/*')
    uploaded_file_count = await func_afklp0ml.put_directory(local_path=str(
        tmp_path / 'folder1'), ignore_file=str(tmp_path / '.prefectignore'))
    assert uploaded_file_count == 2
    (tmp_path / 'downloaded_files').mkdir()
    await func_afklp0ml.get_directory(local_path=str(tmp_path /
        'downloaded_files'))
    assert (tmp_path / 'downloaded_files' / 'file3.txt').exists()
    assert (tmp_path / 'downloaded_files' / 'file4.txt').exists()
    assert not (tmp_path / 'downloaded_files' / 'folder2').exists()
    assert not (tmp_path / 'downloaded_files' / 'folder2' / 'file5.txt'
        ).exists()


async def func_45ue9lsu(s3_bucket, tmp_path, aws_creds_block):
    (tmp_path / 'file1.txt').write_text('FILE 1')
    (tmp_path / 'file2.txt').write_text('FILE 2')
    (tmp_path / 'folder1').mkdir()
    (tmp_path / 'folder1' / 'file3.txt').write_text('FILE 3')
    (tmp_path / 'folder1' / 'file4.txt').write_text('FILE 4')
    (tmp_path / 'folder1' / 'folder2').mkdir()
    (tmp_path / 'folder1' / 'folder2' / 'file5.txt').write_text('FILE 5')
    uploaded_file_count = await func_afklp0ml.put_directory(local_path=str(
        tmp_path / 'folder1'))
    assert uploaded_file_count == 3
    (tmp_path / 'downloaded_files').mkdir()
    await func_afklp0ml.get_directory(local_path=str(tmp_path /
        'downloaded_files'))
    assert (tmp_path / 'downloaded_files' / 'file3.txt').exists()
    assert (tmp_path / 'downloaded_files' / 'file4.txt').exists()
    assert (tmp_path / 'downloaded_files' / 'folder2' / 'file5.txt').exists()


def func_g8vldlc4(s3_bucket_with_file):
    """Test that read path works in a sync context."""
    s3_bucket, key = s3_bucket_with_file
    content = func_afklp0ml.read_path(key)
    assert content == b'hello'


def func_zi462tgo(s3_bucket):
    """Test that write path works in a sync context."""
    key = func_afklp0ml.write_path('test.txt', content=b'hello')
    content = func_afklp0ml.read_path(key)
    assert content == b'hello'


def func_zl4wjd16(s3_bucket):
    assert func_afklp0ml._resolve_path('') == ''


class TestS3Bucket:

    @pytest.fixture(params=[AwsCredentials(), MinIOCredentials(
        minio_root_user='root', minio_root_password='password')])
    def func_20rb7fny(self, request):
        with mock_s3():
            yield request.param

    @pytest.fixture
    def func_m1saxdrl(self, credentials, bucket):
        _s3_bucket = S3Bucket(bucket_name='bucket', credentials=credentials)
        return _s3_bucket

    @pytest.fixture
    def func_3yv1gol4(self, credentials, bucket_2):
        _s3_bucket = S3Bucket(bucket_name='bucket_2', credentials=
            credentials, bucket_folder='subfolder')
        return _s3_bucket

    @pytest.fixture
    def func_0l0dytz8(self, s3_bucket_empty, object):
        _s3_bucket_with_object = s3_bucket_empty
        return _s3_bucket_with_object

    @pytest.fixture
    def func_zgdftjee(self, s3_bucket_2_empty):
        _s3_bucket_with_object = s3_bucket_2_empty
        func_3yv1gol4.write_path('object', content=b'TEST')
        return _s3_bucket_with_object

    @pytest.fixture
    def func_s7li170z(self, s3_bucket_with_object, object_in_folder):
        _s3_bucket_with_objects = s3_bucket_with_object
        return _s3_bucket_with_objects

    @pytest.fixture
    def func_om2fj2c8(self, s3_bucket_with_objects, objects_in_folder):
        _s3_bucket_with_multiple_objects = s3_bucket_with_objects
        return _s3_bucket_with_multiple_objects

    def func_ojcvhxr5(self, credentials):
        s3_bucket = S3Bucket(bucket_name='bucket', credentials=credentials)
        s3_bucket_parsed = S3Bucket.model_validate({'bucket_name': 'bucket',
            'credentials': dict(credentials)})
        assert isinstance(s3_bucket.credentials, type(credentials))
        assert isinstance(s3_bucket_parsed.credentials, type(credentials))

    @pytest.mark.parametrize('client_parameters', aws_clients[-1:],
        indirect=True)
    def func_2vhe821a(self, s3_bucket_empty, client_parameters):
        assert func_m1saxdrl.list_objects() == []

    @pytest.mark.parametrize('client_parameters', aws_clients[-1:],
        indirect=True)
    def func_2czkciky(self, s3_bucket_with_object, client_parameters):
        objects = func_0l0dytz8.list_objects()
        assert len(objects) == 1
        assert [object['Key'] for object in objects] == ['object']

    @pytest.mark.parametrize('client_parameters', aws_clients[-1:],
        indirect=True)
    def func_1bb1jjt3(self, s3_bucket_with_objects, client_parameters):
        objects = func_s7li170z.list_objects()
        assert len(objects) == 2
        assert [object['Key'] for object in objects] == ['folder/object',
            'object']

    @pytest.mark.parametrize('client_parameters', aws_clients[-1:],
        indirect=True)
    def func_plmw33r9(self, s3_bucket_with_similar_objects, client_parameters):
        objects = func_om2fj2c8.list_objects('folder/object/')
        assert len(objects) == 2
        assert [object['Key'] for object in objects] == ['folder/object/bar',
            'folder/object/foo']
        objects = func_om2fj2c8.list_objects('folder')
        assert len(objects) == 5
        assert [object['Key'] for object in objects] == ['folder/object',
            'folder/object/bar', 'folder/object/foo', 'folderobject/bar',
            'folderobject/foo']

    @pytest.mark.parametrize('to_path', [Path('to_path'), 'to_path', None])
    @pytest.mark.parametrize('client_parameters', aws_clients[-1:],
        indirect=True)
    def func_bydddft5(self, s3_bucket_with_object, to_path,
        client_parameters, tmp_path):
        os.chdir(tmp_path)
        func_0l0dytz8.download_object_to_path('object', to_path)
        if to_path is None:
            to_path = tmp_path / 'object'
        to_path = Path(to_path)
        assert to_path.read_text() == 'TEST'

    @pytest.mark.parametrize('client_parameters', aws_clients[-1:],
        indirect=True)
    def func_9d549htd(self, s3_bucket_with_object, client_parameters, tmp_path
        ):
        to_path = tmp_path / 'object'
        with open(to_path, 'wb') as f:
            func_0l0dytz8.download_object_to_file_object('object', f)
        assert to_path.read_text() == 'TEST'

    @pytest.mark.parametrize('to_path', [Path('to_path'), 'to_path', None])
    @pytest.mark.parametrize('client_parameters', aws_clients[-1:],
        indirect=True)
    def func_3vwjlusy(self, s3_bucket_with_objects, client_parameters,
        tmp_path, to_path):
        os.chdir(tmp_path)
        func_s7li170z.download_folder_to_path('folder', to_path)
        if to_path is None:
            to_path = ''
        to_path = Path(to_path)
        assert (to_path / 'object').read_text() == 'TEST OBJECT IN FOLDER'

    @pytest.mark.parametrize('client_parameters', aws_clients[-1:],
        indirect=True)
    def func_fouwncbw(self, s3_bucket_empty, client_parameters, tmp_path):
        """regression test for https://github.com/PrefectHQ/prefect/issues/12848"""
        s3_bucket_empty.bucket_folder = 'some_folder'
        test_content = b'This is a test file.'
        func_m1saxdrl.upload_from_file_object(io.BytesIO(test_content),
            to_path='testing.txt')
        objects = func_m1saxdrl.list_objects()
        assert len(objects) == 1
        assert objects[0]['Key'] == 'some_folder/testing.txt'
        download_path = tmp_path / 'downloaded_test_file.txt'
        func_m1saxdrl.download_object_to_path('testing.txt', download_path)
        assert download_path.read_bytes() == test_content
        s3_bucket_empty.credentials.get_s3_client().delete_object(Bucket=
            s3_bucket_empty.bucket_name, Key=func_m1saxdrl.
            _join_bucket_folder('testing.txt'))

    @pytest.mark.parametrize('to_path', ['to_path', None])
    @pytest.mark.parametrize('client_parameters', aws_clients[-1:],
        indirect=True)
    def func_ox6dgae6(self, s3_bucket_2_with_object, s3_bucket_empty,
        client_parameters, to_path):
        path = func_m1saxdrl.stream_from(s3_bucket_2_with_object, 'object',
            to_path)
        data = func_m1saxdrl.read_path(path)
        assert data == b'TEST'

    @pytest.mark.parametrize('to_path', ['new_object', None])
    @pytest.mark.parametrize('client_parameters', aws_clients[-1:],
        indirect=True)
    def func_mmr18vrc(self, s3_bucket_empty, client_parameters, tmp_path,
        to_path):
        from_path = tmp_path / 'new_object'
        from_path.write_text('NEW OBJECT')
        func_m1saxdrl.upload_from_path(from_path, to_path)
        with io.BytesIO() as buf:
            func_m1saxdrl.download_object_to_file_object('new_object', buf)
            buf.seek(0)
            output = buf.read()
        assert output == b'NEW OBJECT'

    @pytest.mark.parametrize('client_parameters', aws_clients[-1:],
        indirect=True)
    def func_8mg7utmr(self, s3_bucket_empty, client_parameters, tmp_path):
        with open(tmp_path / 'hello', 'wb') as f:
            f.write(b'NEW OBJECT')
        with open(tmp_path / 'hello', 'rb') as f:
            func_m1saxdrl.upload_from_file_object(f, 'new_object')
        with io.BytesIO() as buf:
            func_m1saxdrl.download_object_to_file_object('new_object', buf)
            buf.seek(0)
            output = buf.read()
        assert output == b'NEW OBJECT'

    @pytest.mark.parametrize('client_parameters', aws_clients[-1:],
        indirect=True)
    def func_3k4u0d06(self, s3_bucket_empty, client_parameters, tmp_path,
        caplog):
        from_path = tmp_path / 'new_object'
        from_path.write_text('NEW OBJECT')
        other_path = tmp_path / 'other_object'
        other_path.write_text('OTHER OBJECT')
        folder_dir = tmp_path / 'folder'
        folder_dir.mkdir()
        folder_path = folder_dir / 'other_object'
        folder_path.write_text('FOLDER OBJECT')
        func_m1saxdrl.upload_from_folder(tmp_path)
        new_from_path = tmp_path / 'downloaded_new_object'
        func_m1saxdrl.download_object_to_path('new_object', new_from_path)
        assert new_from_path.read_text() == 'NEW OBJECT'
        new_other_path = tmp_path / 'downloaded_other_object'
        func_m1saxdrl.download_object_to_path('other_object', new_other_path)
        assert new_other_path.read_text() == 'OTHER OBJECT'
        new_folder_path = tmp_path / 'downloaded_folder_object'
        func_m1saxdrl.download_object_to_path('folder/other_object',
            new_folder_path)
        assert new_folder_path.read_text() == 'FOLDER OBJECT'
        empty_folder = tmp_path / 'empty_folder'
        empty_folder.mkdir()
        func_m1saxdrl.upload_from_folder(empty_folder)
        for record in caplog.records:
            if 'No files were uploaded from {empty_folder}':
                break
        else:
            raise AssertionError('Files did upload')

    @pytest.mark.parametrize('client_parameters', aws_clients[-1:],
        indirect=True)
    def func_ro4v2v75(self, s3_bucket_with_object, s3_bucket_2_empty):
        func_0l0dytz8.copy_object('object', 'object_copy_1')
        assert func_0l0dytz8.read_path('object_copy_1') == b'TEST'
        func_0l0dytz8.copy_object('object', 'folder/object_copy_2')
        assert func_0l0dytz8.read_path('folder/object_copy_2') == b'TEST'
        func_0l0dytz8.copy_object('object', func_3yv1gol4._resolve_path(
            'object_copy_3'), to_bucket='bucket_2')
        assert func_3yv1gol4.read_path('object_copy_3') == b'TEST'
        func_0l0dytz8.copy_object('object', 'object_copy_4', s3_bucket_2_empty)
        assert func_3yv1gol4.read_path('object_copy_4') == b'TEST'

    @pytest.mark.parametrize('client_parameters', aws_clients[-1:],
        indirect=True)
    @pytest.mark.parametrize('to_bucket, bucket_folder, expected_path', [(
        None, None, 'object'), (None, 'subfolder', 'subfolder/object'), (
        'bucket_2', None, 'object'), (None, None, 'object'), (None,
        'subfolder', 'subfolder/object'), ('bucket_2', None, 'object')])
    def func_6flp5x21(self, s3_bucket_with_object, s3_bucket_2_empty,
        to_bucket, bucket_folder, expected_path):
        if to_bucket is None:
            to_bucket = s3_bucket_2_empty
            if bucket_folder is not None:
                to_bucket.bucket_folder = bucket_folder
            else:
                to_bucket.bucket_folder = None
        key = func_0l0dytz8.copy_object('object', 'object', to_bucket=to_bucket
            )
        assert key == expected_path

    @pytest.mark.parametrize('client_parameters', aws_clients[-1:],
        indirect=True)
    def func_b0d1cnaa(self, s3_bucket_with_object):
        func_0l0dytz8.move_object('object', 'object_copy_1')
        assert func_0l0dytz8.read_path('object_copy_1') == b'TEST'
        with pytest.raises(ClientError):
            assert func_0l0dytz8.read_path('object') == b'TEST'

    @pytest.mark.parametrize('client_parameters', aws_clients[-1:],
        indirect=True)
    def func_gunic4tc(self, s3_bucket_with_object):
        with pytest.raises(ClientError):
            func_0l0dytz8.move_object('object', 'object_copy_1', to_bucket=
                'nonexistent-bucket')
        assert func_0l0dytz8.read_path('object') == b'TEST'

    @pytest.mark.parametrize('client_parameters', aws_clients[-1:],
        indirect=True)
    def func_k46zti02(self, s3_bucket_with_object):
        with pytest.raises(ClientError):
            func_0l0dytz8.move_object('object', 'object')
        assert func_0l0dytz8.read_path('object') == b'TEST'

    @pytest.mark.parametrize('client_parameters', aws_clients[-1:],
        indirect=True)
    def func_21sykjuz(self, s3_bucket_with_object, s3_bucket_2_empty):
        func_0l0dytz8.move_object('object', 'object_copy_1', to_bucket=
            s3_bucket_2_empty)
        assert func_3yv1gol4.read_path('object_copy_1') == b'TEST'
        with pytest.raises(ClientError):
            assert func_0l0dytz8.read_path('object') == b'TEST'

    @pytest.mark.parametrize('client_parameters', aws_clients[-1:],
        indirect=True)
    @pytest.mark.parametrize('to_bucket, bucket_folder, expected_path', [(
        None, None, 'object'), (None, 'subfolder', 'subfolder/object'), (
        'bucket_2', None, 'object'), (None, None, 'object'), (None,
        'subfolder', 'subfolder/object'), ('bucket_2', None, 'object')])
    def func_1easnil7(self, s3_bucket_with_object, s3_bucket_2_empty,
        to_bucket, bucket_folder, expected_path):
        if to_bucket is None:
            to_bucket = s3_bucket_2_empty
            if bucket_folder is not None:
                to_bucket.bucket_folder = bucket_folder
            else:
                to_bucket.bucket_folder = None
        key = func_0l0dytz8.move_object('object', 'object', to_bucket=to_bucket
            )
        assert key == expected_path

    def func_vrw1u3na(self):
        S3Bucket(bucket_name='round-trip-bucket').save('round-tripper')
        loaded = S3Bucket.load('round-tripper')
        assert hasattr(loaded.credentials, 'aws_access_key_id'
            ), '`credentials` were not properly initialized'

    @pytest.mark.parametrize('client_parameters', [pytest.param(
        'aws_client_parameters_custom_endpoint', marks=pytest.mark.
        is_public(False)), pytest.param(
        'aws_client_parameters_custom_endpoint', marks=pytest.mark.
        is_public(True)), pytest.param('aws_client_parameters_empty', marks
        =pytest.mark.is_public(False)), pytest.param(
        'aws_client_parameters_empty', marks=pytest.mark.is_public(True)),
        pytest.param('aws_client_parameters_public_bucket', marks=[pytest.
        mark.is_public(False), pytest.mark.xfail(reason=
        'Bucket is not a public one')]), pytest.param(
        'aws_client_parameters_public_bucket', marks=pytest.mark.is_public(
        True))], indirect=True)
    async def func_es83ddxc(self, object, client_parameters, aws_credentials):

        @flow
        async def func_fo3hlkj0():
            return await adownload_from_bucket(bucket='bucket', key=
                'object', aws_credentials=aws_credentials,
                aws_client_parameters=client_parameters)
        result = await func_fo3hlkj0()
        assert result == b'TEST'

    @pytest.mark.parametrize('client_parameters', aws_clients, indirect=True)
    async def func_h28wcaw9(self, object, object_in_folder,
        client_parameters, aws_credentials):

        @flow
        async def func_fo3hlkj0():
            return await alist_objects(bucket='bucket', aws_credentials=
                aws_credentials, aws_client_parameters=client_parameters)
        objects = await func_fo3hlkj0()
        assert len(objects) == 2
        assert [object['Key'] for object in objects] == ['folder/object',
            'object']

    @pytest.mark.parametrize('client_parameters', aws_clients, indirect=True)
    async def func_f3fltsrp(self, object, bucket, bucket_2, aws_credentials):

        def func_fqsspnl3(bucket, key):
            stream = io.BytesIO()
            func_5nqqhigu.download_fileobj(key, stream)
            stream.seek(0)
            return stream.read()

        @flow
        async def func_fo3hlkj0():
            await acopy_objects(source_path='object', target_path=
                'subfolder/new_object', source_bucket_name='bucket',
                aws_credentials=aws_credentials, target_bucket_name='bucket_2')
        await func_fo3hlkj0()
        assert func_fqsspnl3(bucket_2, 'subfolder/new_object') == b'TEST'

    @pytest.mark.parametrize('client_parameters', aws_clients, indirect=True)
    async def func_zs01xzqv(self, object, bucket, bucket_2, aws_credentials):

        def func_fqsspnl3(bucket, key):
            stream = io.BytesIO()
            func_5nqqhigu.download_fileobj(key, stream)
            stream.seek(0)
            return stream.read()

        @flow
        async def func_fo3hlkj0():
            await amove_objects(source_path='object', target_path=
                'moved_object', source_bucket_name='bucket',
                target_bucket_name='bucket_2', aws_credentials=aws_credentials)
        await func_fo3hlkj0()
        assert func_fqsspnl3(bucket_2, 'moved_object') == b'TEST'
        with pytest.raises(ClientError):
            func_fqsspnl3(bucket, 'object')
