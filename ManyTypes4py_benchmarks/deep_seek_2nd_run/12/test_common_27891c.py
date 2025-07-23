"""
Tests for the pandas.io.common functionalities
"""
import codecs
import errno
from functools import partial
from io import BytesIO, StringIO, UnsupportedOperation
import mmap
import os
from pathlib import Path
import pickle
import tempfile
from typing import Any, Callable, Dict, IO, List, Optional, Tuple, Type, Union
import numpy as np
import pytest
from pandas.compat import WASM, is_platform_windows
from pandas.compat.pyarrow import pa_version_under19p0
import pandas.util._test_decorators as td
import pandas as pd
import pandas._testing as tm
import pandas.io.common as icom

pytestmark = pytest.mark.filterwarnings('ignore:Passing a BlockManager to DataFrame:DeprecationWarning')

class CustomFSPath:
    """For testing fspath on unknown objects"""

    def __init__(self, path: str) -> None:
        self.path = path

    def __fspath__(self) -> str:
        return self.path

HERE: str = os.path.abspath(os.path.dirname(__file__))

class TestCommonIOCapabilities:
    data1: str = 'index,A,B,C,D\nfoo,2,3,4,5\nbar,7,8,9,10\nbaz,12,13,14,15\nqux,12,13,14,15\nfoo2,12,13,14,15\nbar2,12,13,14,15\n'

    def test_expand_user(self) -> None:
        filename: str = '~/sometest'
        expanded_name: str = icom._expand_user(filename)
        assert expanded_name != filename
        assert os.path.isabs(expanded_name)
        assert os.path.expanduser(filename) == expanded_name

    def test_expand_user_normal_path(self) -> None:
        filename: str = '/somefolder/sometest'
        expanded_name: str = icom._expand_user(filename)
        assert expanded_name == filename
        assert os.path.expanduser(filename) == expanded_name

    def test_stringify_path_pathlib(self) -> None:
        rel_path: str = icom.stringify_path(Path('.'))
        assert rel_path == '.'
        redundant_path: str = icom.stringify_path(Path('foo//bar'))
        assert redundant_path == os.path.join('foo', 'bar')

    def test_stringify_path_fspath(self) -> None:
        p: CustomFSPath = CustomFSPath('foo/bar.csv')
        result: str = icom.stringify_path(p)
        assert result == 'foo/bar.csv'

    def test_stringify_file_and_path_like(self) -> None:
        fsspec = pytest.importorskip('fsspec')
        with tm.ensure_clean() as path:
            with fsspec.open(f'file://{path}', mode='wb') as fsspec_obj:
                assert fsspec_obj == icom.stringify_path(fsspec_obj)

    @pytest.mark.parametrize('path_type', [str, CustomFSPath, Path])
    def test_infer_compression_from_path(self, compression_format: Tuple[str, str], path_type: Type[Union[str, CustomFSPath, Path]]) -> None:
        extension, expected = compression_format
        path = path_type('foo/bar.csv' + extension)
        compression: Optional[str] = icom.infer_compression(path, compression='infer')
        assert compression == expected

    @pytest.mark.parametrize('path_type', [str, CustomFSPath, Path])
    def test_get_handle_with_path(self, path_type: Type[Union[str, CustomFSPath, Path]]) -> None:
        with tempfile.TemporaryDirectory(dir=Path.home()) as tmp:
            filename = path_type('~/' + Path(tmp).name + '/sometest')
            with icom.get_handle(filename, 'w') as handles:
                assert Path(handles.handle.name).is_absolute()
                assert os.path.expanduser(filename) == handles.handle.name

    def test_get_handle_with_buffer(self) -> None:
        with StringIO() as input_buffer:
            with icom.get_handle(input_buffer, 'r') as handles:
                assert handles.handle == input_buffer
            assert not input_buffer.closed
        assert input_buffer.closed

    def test_bytesiowrapper_returns_correct_bytes(self) -> None:
        data: str = 'a,b,c\n1,2,3\n©,®,®\nLook,a snake,🐍'
        with icom.get_handle(StringIO(data), 'rb', is_text=False) as handles:
            result: bytes = b''
            chunksize: int = 5
            while True:
                chunk: bytes = handles.handle.read(chunksize)
                assert len(chunk) <= chunksize
                if len(chunk) < chunksize:
                    assert len(handles.handle.read()) == 0
                    result += chunk
                    break
                result += chunk
            assert result == data.encode('utf-8')

    def test_get_handle_pyarrow_compat(self) -> None:
        pa_csv = pytest.importorskip('pyarrow.csv')
        data: str = 'a,b,c\n1,2,3\n©,®,®\nLook,a snake,🐍'
        expected: pd.DataFrame = pd.DataFrame({'a': ['1', '©', 'Look'], 'b': ['2', '®', 'a snake'], 'c': ['3', '®', '🐍']})
        s = StringIO(data)
        with icom.get_handle(s, 'rb', is_text=False) as handles:
            df: pd.DataFrame = pa_csv.read_csv(handles.handle).to_pandas()
            if pa_version_under19p0:
                expected = expected.astype('object')
            tm.assert_frame_equal(df, expected)
            assert not s.closed

    def test_iterator(self) -> None:
        with pd.read_csv(StringIO(self.data1), chunksize=1) as reader:
            result: pd.DataFrame = pd.concat(reader, ignore_index=True)
        expected: pd.DataFrame = pd.read_csv(StringIO(self.data1))
        tm.assert_frame_equal(result, expected)
        with pd.read_csv(StringIO(self.data1), chunksize=1) as it:
            first: pd.DataFrame = next(it)
            tm.assert_frame_equal(first, expected.iloc[[0]])
            tm.assert_frame_equal(pd.concat(it), expected.iloc[1:])

    @pytest.mark.skipif(WASM, reason='limited file system access on WASM')
    @pytest.mark.parametrize('reader, module, error_class, fn_ext', [(pd.read_csv, 'os', FileNotFoundError, 'csv'), (pd.read_fwf, 'os', FileNotFoundError, 'txt'), (pd.read_excel, 'xlrd', FileNotFoundError, 'xlsx'), (pd.read_feather, 'pyarrow', OSError, 'feather'), (pd.read_hdf, 'tables', FileNotFoundError, 'h5'), (pd.read_stata, 'os', FileNotFoundError, 'dta'), (pd.read_sas, 'os', FileNotFoundError, 'sas7bdat'), (pd.read_json, 'os', FileNotFoundError, 'json'), (pd.read_pickle, 'os', FileNotFoundError, 'pickle')])
    def test_read_non_existent(self, reader: Callable[..., Any], module: str, error_class: Type[Exception], fn_ext: str) -> None:
        pytest.importorskip(module)
        path: str = os.path.join(HERE, 'data', 'does_not_exist.' + fn_ext)
        msg1: str = f"File (b')?.+does_not_exist\\.{fn_ext}'? does not exist"
        msg2: str = f"\\[Errno 2\\] No such file or directory: '.+does_not_exist\\.{fn_ext}'"
        msg3: str = 'Expected object or value'
        msg4: str = 'path_or_buf needs to be a string file path or file-like'
        msg5: str = f"\\[Errno 2\\] File .+does_not_exist\\.{fn_ext} does not exist: '.+does_not_exist\\.{fn_ext}'"
        msg6: str = f"\\[Errno 2\\] 没有那个文件或目录: '.+does_not_exist\\.{fn_ext}'"
        msg7: str = f"\\[Errno 2\\] File o directory non esistente: '.+does_not_exist\\.{fn_ext}'"
        msg8: str = f'Failed to open local file.+does_not_exist\\.{fn_ext}'
        with pytest.raises(error_class, match=f'({msg1}|{msg2}|{msg3}|{msg4}|{msg5}|{msg6}|{msg7}|{msg8})'):
            reader(path)

    @pytest.mark.parametrize('method, module, error_class, fn_ext', [(pd.DataFrame.to_csv, 'os', OSError, 'csv'), (pd.DataFrame.to_html, 'os', OSError, 'html'), (pd.DataFrame.to_excel, 'xlrd', OSError, 'xlsx'), (pd.DataFrame.to_feather, 'pyarrow', OSError, 'feather'), (pd.DataFrame.to_parquet, 'pyarrow', OSError, 'parquet'), (pd.DataFrame.to_stata, 'os', OSError, 'dta'), (pd.DataFrame.to_json, 'os', OSError, 'json'), (pd.DataFrame.to_pickle, 'os', OSError, 'pickle')])
    def test_write_missing_parent_directory(self, method: Callable[..., Any], module: str, error_class: Type[Exception], fn_ext: str) -> None:
        pytest.importorskip(module)
        dummy_frame: pd.DataFrame = pd.DataFrame({'a': [1, 2, 3], 'b': [2, 3, 4], 'c': [3, 4, 5]})
        path: str = os.path.join(HERE, 'data', 'missing_folder', 'does_not_exist.' + fn_ext)
        with pytest.raises(error_class, match='Cannot save file into a non-existent directory: .*missing_folder'):
            method(dummy_frame, path)

    @pytest.mark.skipif(WASM, reason='limited file system access on WASM')
    @pytest.mark.parametrize('reader, module, error_class, fn_ext', [(pd.read_csv, 'os', FileNotFoundError, 'csv'), (pd.read_table, 'os', FileNotFoundError, 'csv'), (pd.read_fwf, 'os', FileNotFoundError, 'txt'), (pd.read_excel, 'xlrd', FileNotFoundError, 'xlsx'), (pd.read_feather, 'pyarrow', OSError, 'feather'), (pd.read_hdf, 'tables', FileNotFoundError, 'h5'), (pd.read_stata, 'os', FileNotFoundError, 'dta'), (pd.read_sas, 'os', FileNotFoundError, 'sas7bdat'), (pd.read_json, 'os', FileNotFoundError, 'json'), (pd.read_pickle, 'os', FileNotFoundError, 'pickle')])
    def test_read_expands_user_home_dir(self, reader: Callable[..., Any], module: str, error_class: Type[Exception], fn_ext: str, monkeypatch: pytest.MonkeyPatch) -> None:
        pytest.importorskip(module)
        path: str = os.path.join('~', 'does_not_exist.' + fn_ext)
        monkeypatch.setattr(icom, '_expand_user', lambda x: os.path.join('foo', x))
        msg1: str = f"File (b')?.+does_not_exist\\.{fn_ext}'? does not exist"
        msg2: str = f"\\[Errno 2\\] No such file or directory: '.+does_not_exist\\.{fn_ext}'"
        msg3: str = "Unexpected character found when decoding 'false'"
        msg4: str = 'path_or_buf needs to be a string file path or file-like'
        msg5: str = f"\\[Errno 2\\] File .+does_not_exist\\.{fn_ext} does not exist: '.+does_not_exist\\.{fn_ext}'"
        msg6: str = f"\\[Errno 2\\] 没有那个文件或目录: '.+does_not_exist\\.{fn_ext}'"
        msg7: str = f"\\[Errno 2\\] File o directory non esistente: '.+does_not_exist\\.{fn_ext}'"
        msg8: str = f'Failed to open local file.+does_not_exist\\.{fn_ext}'
        with pytest.raises(error_class, match=f'({msg1}|{msg2}|{msg3}|{msg4}|{msg5}|{msg6}|{msg7}|{msg8})'):
            reader(path)

    @pytest.mark.parametrize('reader, module, path', [(pd.read_csv, 'os', ('io', 'data', 'csv', 'iris.csv')), (pd.read_table, 'os', ('io', 'data', 'csv', 'iris.csv')), (pd.read_fwf, 'os', ('io', 'data', 'fixed_width', 'fixed_width_format.txt')), (pd.read_excel, 'xlrd', ('io', 'data', 'excel', 'test1.xlsx')), (pd.read_feather, 'pyarrow', ('io', 'data', 'feather', 'feather-0_3_1.feather')), (pd.read_hdf, 'tables', ('io', 'data', 'legacy_hdf', 'pytables_native2.h5')), (pd.read_stata, 'os', ('io', 'data', 'stata', 'stata10_115.dta')), (pd.read_sas, 'os', ('io', 'sas', 'data', 'test1.sas7bdat')), (pd.read_json, 'os', ('io', 'json', 'data', 'tsframe_v012.json')), (pd.read_pickle, 'os', ('io', 'data', 'pickle', 'categorical.0.25.0.pickle'))])
    def test_read_fspath_all(self, reader: Callable[..., Any], module: str, path: Tuple[str, ...], datapath: Callable[..., str]) -> None:
        pytest.importorskip(module)
        path_str: str = datapath(*path)
        mypath: CustomFSPath = CustomFSPath(path_str)
        result: Union[pd.DataFrame, Any] = reader(mypath)
        expected: Union[pd.DataFrame, Any] = reader(path_str)
        if path_str.endswith('.pickle'):
            tm.assert_categorical_equal(result, expected)
        else:
            tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('writer_name, writer_kwargs, module', [('to_csv', {}, 'os'), ('to_excel', {'engine': 'openpyxl'}, 'openpyxl'), ('to_feather', {}, 'pyarrow'), ('to_html', {}, 'os'), ('to_json', {}, 'os'), ('to_latex', {}, 'os'), ('to_pickle', {}, 'os'), ('to_stata', {'time_stamp': pd.to_datetime('2019-01-01 00:00')}, 'os')])
    def test_write_fspath_all(self, writer_name: str, writer_kwargs: Dict[str, Any], module: str) -> None:
        if writer_name in ['to_latex']:
            pytest.importorskip('jinja2')
        p1 = tm.ensure_clean('string')
        p2 = tm.ensure_clean('fspath')
        df: pd.DataFrame = pd.DataFrame({'A': [1, 2]})
        with p1 as string, p2 as fspath:
            pytest.importorskip(module)
            mypath: CustomFSPath = CustomFSPath(fspath)
            writer: Callable[..., Any] = getattr(df, writer_name)
            writer(string, **writer_kwargs)
            writer(mypath, **writer_kwargs)
            with open(string, 'rb') as f_str, open(fspath, 'rb') as f_path:
                if writer_name == 'to_excel':
                    result: pd.DataFrame = pd.read_excel(f_str, **writer_kwargs)
                    expected: pd.DataFrame = pd.read_excel(f_path, **writer_kwargs)
                    tm.assert_frame_equal(result, expected)
                else:
                    result_bytes: bytes = f_str.read()
                    expected_bytes: bytes = f_path.read()
                    assert result_bytes == expected_bytes

    def test_write_fspath_hdf5(self) -> None:
        pytest.importorskip('tables')
        df: pd.DataFrame = pd.DataFrame({'A': [1, 2]})
        p1 = tm.ensure_clean('string')
        p2 = tm.ensure_clean('fspath')
        with p1 as string, p2 as fspath:
            mypath: CustomFSPath = CustomFSPath(fspath)
            df.to_hdf(mypath, key='bar')
            df.to_hdf(string, key='bar')
            result: pd.DataFrame = pd.read_hdf(fspath, key='bar')
            expected: pd.DataFrame = pd.read_hdf(string, key='bar')
        tm.assert_frame_equal(result, expected)

@pytest.fixture
def mmap_file(datapath: Callable[..., str]) -> str:
    return datapath('io', 'data', 'csv', 'test_mmap.csv')

class TestMMapWrapper:

    @pytest.mark.skipif(WASM, reason='limited file system access on WASM')
    def test_constructor_bad_file(self, mmap_file: str) -> None:
        non_file: StringIO = StringIO('I am not a file')
        non_file.fileno = lambda: -1
        if is_platform_windows():
            msg: str = 'The parameter is incorrect'
            err: Type[Exception] = OSError
        else:
            msg = '[Errno 22]'
            err = mmap