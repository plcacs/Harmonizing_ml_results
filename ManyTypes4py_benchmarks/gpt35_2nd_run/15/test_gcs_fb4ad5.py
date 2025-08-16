from io import BytesIO
import os
import pathlib
import tarfile
import zipfile
import numpy as np
import pytest
from pandas.compat.pyarrow import pa_version_under17p0
from pandas import DataFrame, Index, date_range, read_csv, read_excel, read_json, read_parquet
import pandas._testing as tm
from pandas.util import _test_decorators as td
from typing import Any, Dict, List, Tuple

@pytest.fixture
def gcs_buffer() -> BytesIO:
    """Emulate GCS using a binary buffer."""
    pytest.importorskip('gcsfs')
    fsspec = pytest.importorskip('fsspec')
    gcs_buffer: BytesIO = BytesIO()
    gcs_buffer.close = lambda: True

    class MockGCSFileSystem(fsspec.AbstractFileSystem):

        @staticmethod
        def open(*args: Any, **kwargs: Any) -> BytesIO:
            gcs_buffer.seek(0)
            return gcs_buffer

        def ls(self, path: str, **kwargs: Any) -> List[Dict[str, str]]:
            return [{'name': path, 'type': 'file'}]
    fsspec.register_implementation('gs', MockGCSFileSystem, clobber=True)
    return gcs_buffer

def assert_equal_zip_safe(result: bytes, expected: bytes, compression: str) -> None:
    """
    For zip compression, only compare the CRC-32 checksum of the file contents
    to avoid checking the time-dependent last-modified timestamp which
    in some CI builds is off-by-one

    See https://en.wikipedia.org/wiki/ZIP_(file_format)#File_headers
    """
    if compression == 'zip':
        with zipfile.ZipFile(BytesIO(result)) as exp, zipfile.ZipFile(BytesIO(expected)) as res:
            for res_info, exp_info in zip(res.infolist(), exp.infolist()):
                assert res_info.CRC == exp_info.CRC
    elif compression == 'tar':
        with tarfile.open(fileobj=BytesIO(result)) as tar_exp, tarfile.open(fileobj=BytesIO(expected)) as tar_res:
            for tar_res_info, tar_exp_info in zip(tar_res.getmembers(), tar_exp.getmembers()):
                actual_file = tar_res.extractfile(tar_res_info)
                expected_file = tar_exp.extractfile(tar_exp_info)
                assert (actual_file is None) == (expected_file is None)
                if actual_file is not None and expected_file is not None:
                    assert actual_file.read() == expected_file.read()
    else:
        assert result == expected

@pytest.mark.parametrize('encoding', ['utf-8', 'cp1251'])
def test_to_csv_compression_encoding_gcs(gcs_buffer: BytesIO, compression_only: str, encoding: str, compression_to_extension: Dict[str, str]) -> None:
    """
    Compression and encoding should with GCS.

    GH 35677 (to_csv, compression), GH 26124 (to_csv, encoding), and
    GH 32392 (read_csv, encoding)
    """
    df: DataFrame = DataFrame(1.1 * np.arange(120).reshape((30, 4)), columns=Index(list('ABCD')), index=Index([f'i-{i}' for i in range(30)]))
    compression: Dict[str, Any] = {'method': compression_only}
    if compression_only == 'gzip':
        compression['mtime'] = 1
    buffer: BytesIO = BytesIO()
    df.to_csv(buffer, compression=compression, encoding=encoding, mode='wb')
    path_gcs: str = 'gs://test/test.csv'
    df.to_csv(path_gcs, compression=compression, encoding=encoding)
    res: bytes = gcs_buffer.getvalue()
    expected: bytes = buffer.getvalue()
    assert_equal_zip_safe(res, expected, compression_only)
    read_df: DataFrame = read_csv(path_gcs, index_col=0, compression=compression_only, encoding=encoding)
    tm.assert_frame_equal(df, read_df)
    file_ext: str = compression_to_extension[compression_only]
    compression['method'] = 'infer'
    path_gcs += f'.{file_ext}'
    df.to_csv(path_gcs, compression=compression, encoding=encoding)
    res = gcs_buffer.getvalue()
    expected = buffer.getvalue()
    assert_equal_zip_safe(res, expected, compression_only)
    read_df = read_csv(path_gcs, index_col=0, compression='infer', encoding=encoding)
    tm.assert_frame_equal(df, read_df)

def test_to_parquet_gcs_new_file(monkeypatch: Any, tmpdir: Any) -> None:
    """Regression test for writing to a not-yet-existent GCS Parquet file."""
    pytest.importorskip('fastparquet')
    pytest.importorskip('gcsfs')
    from fsspec import AbstractFileSystem
    df1: DataFrame = DataFrame({'int': [1, 3], 'float': [2.0, np.nan], 'str': ['t', 's'], 'dt': date_range('2018-06-18', periods=2)})

    class MockGCSFileSystem(AbstractFileSystem):

        def open(self, path: str, mode: str = 'r', *args: Any) -> Any:
            if 'w' not in mode:
                raise FileNotFoundError
            return open(os.path.join(tmpdir, 'test.parquet'), mode, encoding='utf-8')
    monkeypatch.setattr('gcsfs.GCSFileSystem', MockGCSFileSystem)
    df1.to_parquet('gs://test/test.csv', index=True, engine='fastparquet', compression=None)

@td.skip_if_installed('gcsfs')
def test_gcs_not_present_exception() -> None:
    with tm.external_error_raised(ImportError):
        read_csv('gs://test/test.csv')
