import shlex
import subprocess
import time
import uuid
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import pandas as pd
import pytest
from pandas.compat import is_ci_environment, is_platform_arm, is_platform_mac, is_platform_windows
import pandas.util._test_decorators as td
import pandas.io.common as icom
from pandas.io.parsers import read_csv


@pytest.fixture
def compression_to_extension() -> Dict[Optional[str], str]:
    return {value: key for key, value in icom.extension_to_compression.items()}


@pytest.fixture
def tips_file(datapath: Callable[..., str]) -> str:
    """Path to the tips dataset"""
    return datapath('io', 'data', 'csv', 'tips.csv')


@pytest.fixture
def jsonl_file(datapath: Callable[..., str]) -> str:
    """Path to a JSONL dataset"""
    return datapath('io', 'parser', 'data', 'items.jsonl')


@pytest.fixture
def salaries_table(datapath: Callable[..., str]) -> pd.DataFrame:
    """DataFrame with the salaries dataset"""
    return read_csv(datapath('io', 'parser', 'data', 'salaries.csv'), sep='\t')


@pytest.fixture
def feather_file(datapath: Callable[..., str]) -> str:
    return datapath('io', 'data', 'feather', 'feather-0_3_1.feather')


@pytest.fixture
def xml_file(datapath: Callable[..., str]) -> str:
    return datapath('io', 'data', 'xml', 'books.xml')


@pytest.fixture
def s3_base(worker_id: str, monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    """
    Fixture for mocking S3 interaction.

    Sets up moto server in separate process locally
    Return url for motoserver/moto CI service
    """
    pytest.importorskip('s3fs')
    pytest.importorskip('boto3')
    monkeypatch.setenv('AWS_ACCESS_KEY_ID', 'foobar_key')
    monkeypatch.setenv('AWS_SECRET_ACCESS_KEY', 'foobar_secret')
    if is_ci_environment():
        if is_platform_arm() or is_platform_mac() or is_platform_windows():
            pytest.skip('S3 tests do not have a corresponding service on Windows or macOS platforms')
        else:
            yield 'http://localhost:5000'
    else:
        requests = pytest.importorskip('requests')
        pytest.importorskip('moto')
        pytest.importorskip('flask')
        worker_id = '5' if worker_id == 'master' else worker_id.lstrip('gw')
        endpoint_port = f'555{worker_id}'
        endpoint_uri = f'http://127.0.0.1:{endpoint_port}/'
        with subprocess.Popen(
            shlex.split(f'moto_server s3 -p {endpoint_port}'),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ) as proc:
            timeout = 5.0
            while timeout > 0:
                try:
                    r = requests.get(endpoint_uri)
                    if r.ok:
                        break
                except Exception:
                    pass
                timeout -= 0.1
                time.sleep(0.1)
            yield endpoint_uri
            proc.terminate()


@pytest.fixture
def s3so(s3_base: str) -> Dict[str, Dict[str, str]]:
    return {'client_kwargs': {'endpoint_url': s3_base}}


@pytest.fixture
def s3_resource(s3_base: str) -> Any:
    import boto3
    s3 = boto3.resource('s3', endpoint_url=s3_base)
    return s3


@pytest.fixture
def s3_public_bucket(s3_resource: Any) -> Iterator[Any]:
    bucket = s3_resource.Bucket(f'pandas-test-{uuid.uuid4()}')
    bucket.create()
    yield bucket
    bucket.objects.delete()
    bucket.delete()


@pytest.fixture
def s3_public_bucket_with_data(
    s3_public_bucket: Any,
    tips_file: str,
    jsonl_file: str,
    feather_file: str,
    xml_file: str,
) -> Any:
    """
    The following datasets
    are loaded.

    - tips.csv
    - tips.csv.gz
    - tips.csv.bz2
    - items.jsonl
    """
    test_s3_files: List[Tuple[str, str]] = [
        ('tips#1.csv', tips_file),
        ('tips.csv', tips_file),
        ('tips.csv.gz', tips_file + '.gz'),
        ('tips.csv.bz2', tips_file + '.bz2'),
        ('items.jsonl', jsonl_file),
        ('simple_dataset.feather', feather_file),
        ('books.xml', xml_file),
    ]
    for s3_key, file_name in test_s3_files:
        with open(file_name, 'rb') as f:
            s3_public_bucket.put_object(Key=s3_key, Body=f)
    return s3_public_bucket


@pytest.fixture
def s3_private_bucket(s3_resource: Any) -> Iterator[Any]:
    bucket = s3_resource.Bucket(f'cant_get_it-{uuid.uuid4()}')
    bucket.create(ACL='private')
    yield bucket
    bucket.objects.delete()
    bucket.delete()


@pytest.fixture
def s3_private_bucket_with_data(
    s3_private_bucket: Any,
    tips_file: str,
    jsonl_file: str,
    feather_file: str,
    xml_file: str,
) -> Any:
    """
    The following datasets
    are loaded.

    - tips.csv
    - tips.csv.gz
    - tips.csv.bz2
    - items.jsonl
    """
    test_s3_files: List[Tuple[str, str]] = [
        ('tips#1.csv', tips_file),
        ('tips.csv', tips_file),
        ('tips.csv.gz', tips_file + '.gz'),
        ('tips.csv.bz2', tips_file + '.bz2'),
        ('items.jsonl', jsonl_file),
        ('simple_dataset.feather', feather_file),
        ('books.xml', xml_file),
    ]
    for s3_key, file_name in test_s3_files:
        with open(file_name, 'rb') as f:
            s3_private_bucket.put_object(Key=s3_key, Body=f)
    return s3_private_bucket


_compression_formats_params: List[Any] = [
    ('.no_compress', None),
    ('', None),
    ('.gz', 'gzip'),
    ('.GZ', 'gzip'),
    ('.bz2', 'bz2'),
    ('.BZ2', 'bz2'),
    ('.zip', 'zip'),
    ('.ZIP', 'zip'),
    ('.xz', 'xz'),
    ('.XZ', 'xz'),
    pytest.param(('.zst', 'zstd'), marks=td.skip_if_no('zstandard')),
    pytest.param(('.ZST', 'zstd'), marks=td.skip_if_no('zstandard')),
]


@pytest.fixture(params=_compression_formats_params[1:])
def compression_format(request: pytest.FixtureRequest) -> Tuple[str, Optional[str]]:
    return request.param


@pytest.fixture(params=_compression_formats_params)
def compression_ext(request: pytest.FixtureRequest) -> str:
    return request.param[0]