"""
Stub file for pandas.io.common test module
"""

from typing import (
    Any,
    AnyStr,
    Callable,
    Dict,
    Generator,
    Iterable,
    List,
    Optional,
    Tuple,
    Union,
)
from pathlib import Path
from io import BytesIO, StringIO
import errno
import mmap
import pickle
import pytest
import numpy as np
import pandas as pd
import pandas._testing as tm
import pandas.io.common as icom

class CustomFSPath:
    def __init__(self, path: str) -> None:
        ...
    
    def __fspath__(self) -> str:
        ...

class TestCommonIOCapabilities:
    data1: str = ...
    
    def test_expand_user(self, filename: str) -> None:
        ...
    
    def test_expand_user_normal_path(self, filename: str) -> None:
        ...
    
    def test_stringify_path_pathlib(self, rel_path: Path) -> None:
        ...
    
    def test_stringify_path_fspath(self, p: CustomFSPath) -> None:
        ...
    
    def test_stringify_file_and_path_like(self) -> None:
        ...
    
    @pytest.mark.parametrize('path_type', [str, CustomFSPath, Path])
    def test_infer_compression_from_path(
        self,
        compression_format: tuple[str, str],
        path_type: type
    ) -> None:
        ...
    
    @pytest.mark.parametrize('path_type', [str, CustomFSPath, Path])
    def test_get_handle_with_path(self, path_type: type) -> None:
        ...
    
    def test_get_handle_with_buffer(self) -> None:
        ...
    
    def test_bytesiowrapper_returns_correct_bytes(self) -> None:
        ...
    
    def test_get_handle_pyarrow_compat(self) -> None:
        ...
    
    def test_iterator(self) -> None:
        ...
    
    @pytest.mark.skipif(icom.WASM, reason: str)
    @pytest.mark.parametrize('reader, module, error_class, fn_ext', [
        (pd.read_csv, 'os', FileNotFoundError, 'csv'),
        (pd.read_fwf, 'os', FileNotFoundError, 'txt'),
        (pd.read_excel, 'xlrd', FileNotFoundError, 'xlsx'),
        (pd.read_feather, 'pyarrow', OSError, 'feather'),
        (pd.read_hdf, 'tables', FileNotFoundError, 'h5'),
        (pd.read_stata, 'os', FileNotFoundError, 'dta'),
        (pd.read_sas, 'os', FileNotFoundError, 'sas7bdat'),
        (pd.read_json, 'os', FileNotFoundError, 'json'),
        (pd.read_pickle, 'os', FileNotFoundError, 'pickle')
    ])
    def test_read_non_existent(
        self,
        reader: Callable,
        module: str,
        error_class: type,
        fn_ext: str
    ) -> None:
        ...
    
    @pytest.mark.parametrize('method, module, error_class, fn_ext', [
        (pd.DataFrame.to_csv, 'os', OSError, 'csv'),
        (pd.DataFrame.to_html, 'os', OSError, 'html'),
        (pd.DataFrame.to_excel, 'xlrd', OSError, 'xlsx'),
        (pd.DataFrame.to_feather, 'pyarrow', OSError, 'feather'),
        (pd.DataFrame.to_parquet, 'pyarrow', OSError, 'parquet'),
        (pd.DataFrame.to_stata, 'os', OSError, 'dta'),
        (pd.DataFrame.to_json, 'os', OSError, 'json'),
        (pd.DataFrame.to_pickle, 'os', OSError, 'pickle')
    ])
    def test_write_missing_parent_directory(
        self,
        method: Callable,
        module: str,
        error_class: type,
        fn_ext: str
    ) -> None:
        ...
    
    @pytest.mark.skipif(icom.WASM, reason: str)
    @pytest.mark.parametrize('reader, module, error_class, fn_ext', [
        (pd.read_csv, 'os', FileNotFoundError, 'csv'),
        (pd.read_table, 'os', FileNotFoundError, 'csv'),
        (pd.read_fwf, 'os', FileNotFoundError, 'txt'),
        (pd.read_excel, 'xlrd', FileNotFoundError, 'xlsx'),
        (pd.read_feather, 'pyarrow', OSError, 'feather'),
        (pd.read_hdf, 'tables', FileNotFoundError, 'h5'),
        (pd.read_stata, 'os', FileNotFoundError, 'dta'),
        (pd.read_sas, 'os', FileNotFoundError, 'sas7bdat'),
        (pd.read_json, 'os', FileNotFoundError, 'json'),
        (pd.read_pickle, 'os', FileNotFoundError, 'pickle')
    ])
    def test_read_expands_user_home_dir(
        self,
        reader: Callable,
        module: str,
        error_class: type,
        fn_ext: str,
        monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ...
    
    @pytest.mark.parametrize('reader, module, path', [
        (pd.read_csv, 'os', ('io', 'data', 'csv', 'iris.csv')),
        (pd.read_table, 'os', ('io', 'data', 'csv', 'iris.csv')),
        (pd.read_fwf, 'os', ('io', 'data', 'fixed_width', 'fixed_width_format.txt')),
        (pd.read_excel, 'xlrd', ('io', 'data', 'excel', 'test1.xlsx')),
        (pd.read_feather, 'pyarrow', ('io', 'data', 'feather', 'feather-0_3_1.feather')),
        (pd.read_hdf, 'tables', ('io', 'data', 'legacy_hdf', 'pytables_native2.h5')),
        (pd.read_stata, 'os', ('io', 'data', 'stata', 'stata10_115.dta')),
        (pd.read_sas, 'os', ('io', 'sas', 'data', 'test1.sas7bdat')),
        (pd.read_json, 'os', ('io', 'json', 'data', 'tsframe_v012.json')),
        (pd.read_pickle, 'os', ('io', 'data', 'pickle', 'categorical.0.25.0.pickle'))
    ])
    def test_read_fspath_all(
        self,
        reader: Callable,
        module: str,
        path: tuple[str, ...],
        datapath: Callable
    ) -> None:
        ...
    
    @pytest.mark.parametrize('writer_name, writer_kwargs, module', [
        ('to_csv', {}, 'os'),
        ('to_excel', {'engine': 'openpyxl'}, 'openpyxl'),
        ('to_feather', {}, 'pyarrow'),
        ('to_html', {}, 'os'),
        ('to_json', {}, 'os'),
        ('to_latex', {}, 'os'),
        ('to_pickle', {}, 'os'),
        ('to_stata', {'time_stamp': pd.to_datetime('2019-01-01 00:00')}, 'os')
    ])
    def test_write_fspath_all(
        self,
        writer_name: str,
        writer_kwargs: dict,
        module: str
    ) -> None:
        ...
    
    def test_write_fspath_hdf5(self) -> None:
        ...
    
@pytest.fixture
def mmap_file(datapath: Callable) -> str:
    ...

class TestMMapWrapper:
    @pytest.mark.skipif(icom.WASM, reason: str)
    def test_constructor_bad_file(self, mmap_file: str) -> None:
        ...
    
    @pytest.mark.skipif(icom.WASM, reason: str)
    def test_next(self, mmap_file: str) -> None:
        ...
    
    def test_unknown_engine(self) -> None:
        ...
    
    def test_binary_mode(self) -> None:
        ...
    
    @pytest.mark.parametrize('encoding', ['utf-16', 'utf-32'])
    @pytest.mark.parametrize('compression_', ['bz2', 'xz'])
    def test_warning_missing_utf_bom(
        self,
        encoding: str,
        compression_: str
    ) -> None:
        ...
    
def test_is_fsspec_url() -> bool:
    ...

@pytest.mark.parametrize('encoding', [None, 'utf-8'])
@pytest.mark.parametrize('format', ['csv', 'json'])
def test_codecs_encoding(
    encoding: Optional[str],
    format: str
) -> None:
    ...

def test_codecs_get_writer_reader() -> None:
    ...

@pytest.mark.parametrize('io_class,mode,msg', [
    (BytesIO, 't', "a bytes-like object is required, not 'str'"),
    (StringIO, 'b', "string argument expected, got 'bytes'")
])
def test_explicit_encoding(
    io_class: type,
    mode: str,
    msg: str
) -> None:
    ...

@pytest.mark.parametrize('encoding_errors', ['strict', 'replace'])
@pytest.mark.parametrize('format', ['csv', 'json'])
def test_encoding_errors(
    encoding_errors: str,
    format: str
) -> None:
    ...

@pytest.mark.parametrize('encoding_errors', [0, None])
def test_encoding_errors_badtype(
    encoding_errors: Union[int, None]
) -> None:
    ...

def test_bad_encdoing_errors() -> None:
    ...

@pytest.mark.skipif(icom.WASM, reason: str)
def test_errno_attribute() -> None:
    ...

def test_fail_mmap() -> None:
    ...

def test_close_on_error() -> None:
    ...

@td.skip_if_no('fsspec', min_version='2023.1.0')
@pytest.mark.parametrize('compression', [None, 'infer'])
def test_read_csv_chained_url_no_error(
    compression: Optional[str]
) -> None:
    ...

@pytest.mark.parametrize('reader', [
    pd.read_csv,
    pd.read_fwf,
    pd.read_excel,
    pd.read_feather,
    pd.read_hdf,
    pd.read_stata,
    pd.read_sas,
    pd.read_json,
    pd.read_pickle
])
def test_pickle_reader(reader: Callable) -> None:
    ...

@td.skip_if_no('pyarrow')
def test_pyarrow_read_csv_datetime_dtype() -> None:
    ...