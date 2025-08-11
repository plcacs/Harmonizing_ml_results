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

    def __init__(self, path: Any) -> None:
        self.path = path

    def __fspath__(self):
        return self.path
HERE = os.path.abspath(os.path.dirname(__file__))

class TestCommonIOCapabilities:
    data1 = 'index,A,B,C,D\nfoo,2,3,4,5\nbar,7,8,9,10\nbaz,12,13,14,15\nqux,12,13,14,15\nfoo2,12,13,14,15\nbar2,12,13,14,15\n'

    def test_expand_user(self) -> None:
        filename = '~/sometest'
        expanded_name = icom._expand_user(filename)
        assert expanded_name != filename
        assert os.path.isabs(expanded_name)
        assert os.path.expanduser(filename) == expanded_name

    def test_expand_user_normal_path(self) -> None:
        filename = '/somefolder/sometest'
        expanded_name = icom._expand_user(filename)
        assert expanded_name == filename
        assert os.path.expanduser(filename) == expanded_name

    def test_stringify_path_pathlib(self) -> None:
        rel_path = icom.stringify_path(Path('.'))
        assert rel_path == '.'
        redundant_path = icom.stringify_path(Path('foo//bar'))
        assert redundant_path == os.path.join('foo', 'bar')

    def test_stringify_path_fspath(self) -> None:
        p = CustomFSPath('foo/bar.csv')
        result = icom.stringify_path(p)
        assert result == 'foo/bar.csv'

    def test_stringify_file_and_path_like(self) -> None:
        fsspec = pytest.importorskip('fsspec')
        with tm.ensure_clean() as path:
            with fsspec.open(f'file://{path}', mode='wb') as fsspec_obj:
                assert fsspec_obj == icom.stringify_path(fsspec_obj)

    @pytest.mark.parametrize('path_type', [str, CustomFSPath, Path])
    def test_infer_compression_from_path(self, compression_format: Any, path_type: Any) -> None:
        extension, expected = compression_format
        path = path_type('foo/bar.csv' + extension)
        compression = icom.infer_compression(path, compression='infer')
        assert compression == expected

    @pytest.mark.parametrize('path_type', [str, CustomFSPath, Path])
    def test_get_handle_with_path(self, path_type: Any) -> None:
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
        data = 'a,b,c\n1,2,3\n©,®,®\nLook,a snake,🐍'
        with icom.get_handle(StringIO(data), 'rb', is_text=False) as handles:
            result = b''
            chunksize = 5
            while True:
                chunk = handles.handle.read(chunksize)
                assert len(chunk) <= chunksize
                if len(chunk) < chunksize:
                    assert len(handles.handle.read()) == 0
                    result += chunk
                    break
                result += chunk
            assert result == data.encode('utf-8')

    def test_get_handle_pyarrow_compat(self) -> None:
        pa_csv = pytest.importorskip('pyarrow.csv')
        data = 'a,b,c\n1,2,3\n©,®,®\nLook,a snake,🐍'
        expected = pd.DataFrame({'a': ['1', '©', 'Look'], 'b': ['2', '®', 'a snake'], 'c': ['3', '®', '🐍']})
        s = StringIO(data)
        with icom.get_handle(s, 'rb', is_text=False) as handles:
            df = pa_csv.read_csv(handles.handle).to_pandas()
            if pa_version_under19p0:
                expected = expected.astype('object')
            tm.assert_frame_equal(df, expected)
            assert not s.closed

    def test_iterator(self) -> None:
        with pd.read_csv(StringIO(self.data1), chunksize=1) as reader:
            result = pd.concat(reader, ignore_index=True)
        expected = pd.read_csv(StringIO(self.data1))
        tm.assert_frame_equal(result, expected)
        with pd.read_csv(StringIO(self.data1), chunksize=1) as it:
            first = next(it)
            tm.assert_frame_equal(first, expected.iloc[[0]])
            tm.assert_frame_equal(pd.concat(it), expected.iloc[1:])

    @pytest.mark.skipif(WASM, reason='limited file system access on WASM')
    @pytest.mark.parametrize('reader, module, error_class, fn_ext', [(pd.read_csv, 'os', FileNotFoundError, 'csv'), (pd.read_fwf, 'os', FileNotFoundError, 'txt'), (pd.read_excel, 'xlrd', FileNotFoundError, 'xlsx'), (pd.read_feather, 'pyarrow', OSError, 'feather'), (pd.read_hdf, 'tables', FileNotFoundError, 'h5'), (pd.read_stata, 'os', FileNotFoundError, 'dta'), (pd.read_sas, 'os', FileNotFoundError, 'sas7bdat'), (pd.read_json, 'os', FileNotFoundError, 'json'), (pd.read_pickle, 'os', FileNotFoundError, 'pickle')])
    def test_read_non_existent(self, reader: Any, module: Any, error_class: Any, fn_ext: Any) -> None:
        pytest.importorskip(module)
        path = os.path.join(HERE, 'data', 'does_not_exist.' + fn_ext)
        msg1 = f"File (b')?.+does_not_exist\\.{fn_ext}'? does not exist"
        msg2 = f"\\[Errno 2\\] No such file or directory: '.+does_not_exist\\.{fn_ext}'"
        msg3 = 'Expected object or value'
        msg4 = 'path_or_buf needs to be a string file path or file-like'
        msg5 = f"\\[Errno 2\\] File .+does_not_exist\\.{fn_ext} does not exist: '.+does_not_exist\\.{fn_ext}'"
        msg6 = f"\\[Errno 2\\] 没有那个文件或目录: '.+does_not_exist\\.{fn_ext}'"
        msg7 = f"\\[Errno 2\\] File o directory non esistente: '.+does_not_exist\\.{fn_ext}'"
        msg8 = f'Failed to open local file.+does_not_exist\\.{fn_ext}'
        with pytest.raises(error_class, match=f'({msg1}|{msg2}|{msg3}|{msg4}|{msg5}|{msg6}|{msg7}|{msg8})'):
            reader(path)

    @pytest.mark.parametrize('method, module, error_class, fn_ext', [(pd.DataFrame.to_csv, 'os', OSError, 'csv'), (pd.DataFrame.to_html, 'os', OSError, 'html'), (pd.DataFrame.to_excel, 'xlrd', OSError, 'xlsx'), (pd.DataFrame.to_feather, 'pyarrow', OSError, 'feather'), (pd.DataFrame.to_parquet, 'pyarrow', OSError, 'parquet'), (pd.DataFrame.to_stata, 'os', OSError, 'dta'), (pd.DataFrame.to_json, 'os', OSError, 'json'), (pd.DataFrame.to_pickle, 'os', OSError, 'pickle')])
    def test_write_missing_parent_directory(self, method: Any, module: Any, error_class: Any, fn_ext: Any) -> None:
        pytest.importorskip(module)
        dummy_frame = pd.DataFrame({'a': [1, 2, 3], 'b': [2, 3, 4], 'c': [3, 4, 5]})
        path = os.path.join(HERE, 'data', 'missing_folder', 'does_not_exist.' + fn_ext)
        with pytest.raises(error_class, match='Cannot save file into a non-existent directory: .*missing_folder'):
            method(dummy_frame, path)

    @pytest.mark.skipif(WASM, reason='limited file system access on WASM')
    @pytest.mark.parametrize('reader, module, error_class, fn_ext', [(pd.read_csv, 'os', FileNotFoundError, 'csv'), (pd.read_table, 'os', FileNotFoundError, 'csv'), (pd.read_fwf, 'os', FileNotFoundError, 'txt'), (pd.read_excel, 'xlrd', FileNotFoundError, 'xlsx'), (pd.read_feather, 'pyarrow', OSError, 'feather'), (pd.read_hdf, 'tables', FileNotFoundError, 'h5'), (pd.read_stata, 'os', FileNotFoundError, 'dta'), (pd.read_sas, 'os', FileNotFoundError, 'sas7bdat'), (pd.read_json, 'os', FileNotFoundError, 'json'), (pd.read_pickle, 'os', FileNotFoundError, 'pickle')])
    def test_read_expands_user_home_dir(self, reader: Any, module: Any, error_class: Any, fn_ext: Any, monkeypatch: Any) -> None:
        pytest.importorskip(module)
        path = os.path.join('~', 'does_not_exist.' + fn_ext)
        monkeypatch.setattr(icom, '_expand_user', lambda x: os.path.join('foo', x))
        msg1 = f"File (b')?.+does_not_exist\\.{fn_ext}'? does not exist"
        msg2 = f"\\[Errno 2\\] No such file or directory: '.+does_not_exist\\.{fn_ext}'"
        msg3 = "Unexpected character found when decoding 'false'"
        msg4 = 'path_or_buf needs to be a string file path or file-like'
        msg5 = f"\\[Errno 2\\] File .+does_not_exist\\.{fn_ext} does not exist: '.+does_not_exist\\.{fn_ext}'"
        msg6 = f"\\[Errno 2\\] 没有那个文件或目录: '.+does_not_exist\\.{fn_ext}'"
        msg7 = f"\\[Errno 2\\] File o directory non esistente: '.+does_not_exist\\.{fn_ext}'"
        msg8 = f'Failed to open local file.+does_not_exist\\.{fn_ext}'
        with pytest.raises(error_class, match=f'({msg1}|{msg2}|{msg3}|{msg4}|{msg5}|{msg6}|{msg7}|{msg8})'):
            reader(path)

    @pytest.mark.parametrize('reader, module, path', [(pd.read_csv, 'os', ('io', 'data', 'csv', 'iris.csv')), (pd.read_table, 'os', ('io', 'data', 'csv', 'iris.csv')), (pd.read_fwf, 'os', ('io', 'data', 'fixed_width', 'fixed_width_format.txt')), (pd.read_excel, 'xlrd', ('io', 'data', 'excel', 'test1.xlsx')), (pd.read_feather, 'pyarrow', ('io', 'data', 'feather', 'feather-0_3_1.feather')), (pd.read_hdf, 'tables', ('io', 'data', 'legacy_hdf', 'pytables_native2.h5')), (pd.read_stata, 'os', ('io', 'data', 'stata', 'stata10_115.dta')), (pd.read_sas, 'os', ('io', 'sas', 'data', 'test1.sas7bdat')), (pd.read_json, 'os', ('io', 'json', 'data', 'tsframe_v012.json')), (pd.read_pickle, 'os', ('io', 'data', 'pickle', 'categorical.0.25.0.pickle'))])
    def test_read_fspath_all(self, reader: Any, module: Any, path: Any, datapath: Any) -> None:
        pytest.importorskip(module)
        path = datapath(*path)
        mypath = CustomFSPath(path)
        result = reader(mypath)
        expected = reader(path)
        if path.endswith('.pickle'):
            tm.assert_categorical_equal(result, expected)
        else:
            tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('writer_name, writer_kwargs, module', [('to_csv', {}, 'os'), ('to_excel', {'engine': 'openpyxl'}, 'openpyxl'), ('to_feather', {}, 'pyarrow'), ('to_html', {}, 'os'), ('to_json', {}, 'os'), ('to_latex', {}, 'os'), ('to_pickle', {}, 'os'), ('to_stata', {'time_stamp': pd.to_datetime('2019-01-01 00:00')}, 'os')])
    def test_write_fspath_all(self, writer_name: Any, writer_kwargs: Any, module: Any) -> None:
        if writer_name in ['to_latex']:
            pytest.importorskip('jinja2')
        p1 = tm.ensure_clean('string')
        p2 = tm.ensure_clean('fspath')
        df = pd.DataFrame({'A': [1, 2]})
        with p1 as string, p2 as fspath:
            pytest.importorskip(module)
            mypath = CustomFSPath(fspath)
            writer = getattr(df, writer_name)
            writer(string, **writer_kwargs)
            writer(mypath, **writer_kwargs)
            with open(string, 'rb') as f_str, open(fspath, 'rb') as f_path:
                if writer_name == 'to_excel':
                    result = pd.read_excel(f_str, **writer_kwargs)
                    expected = pd.read_excel(f_path, **writer_kwargs)
                    tm.assert_frame_equal(result, expected)
                else:
                    result = f_str.read()
                    expected = f_path.read()
                    assert result == expected

    def test_write_fspath_hdf5(self) -> None:
        pytest.importorskip('tables')
        df = pd.DataFrame({'A': [1, 2]})
        p1 = tm.ensure_clean('string')
        p2 = tm.ensure_clean('fspath')
        with p1 as string, p2 as fspath:
            mypath = CustomFSPath(fspath)
            df.to_hdf(mypath, key='bar')
            df.to_hdf(string, key='bar')
            result = pd.read_hdf(fspath, key='bar')
            expected = pd.read_hdf(string, key='bar')
        tm.assert_frame_equal(result, expected)

@pytest.fixture
def mmap_file(datapath: Any):
    return datapath('io', 'data', 'csv', 'test_mmap.csv')

class TestMMapWrapper:

    @pytest.mark.skipif(WASM, reason='limited file system access on WASM')
    def test_constructor_bad_file(self, mmap_file: Any) -> None:
        non_file = StringIO('I am not a file')
        non_file.fileno = lambda: -1
        if is_platform_windows():
            msg = 'The parameter is incorrect'
            err = OSError
        else:
            msg = '[Errno 22]'
            err = mmap.error
        with pytest.raises(err, match=msg):
            icom._maybe_memory_map(non_file, True)
        with open(mmap_file, encoding='utf-8') as target:
            pass
        msg = 'I/O operation on closed file'
        with pytest.raises(ValueError, match=msg):
            icom._maybe_memory_map(target, True)

    @pytest.mark.skipif(WASM, reason='limited file system access on WASM')
    def test_next(self, mmap_file: Any) -> None:
        with open(mmap_file, encoding='utf-8') as target:
            lines = target.readlines()
            with icom.get_handle(target, 'r', is_text=True, memory_map=True) as wrappers:
                wrapper = wrappers.handle
                assert isinstance(wrapper.buffer.buffer, mmap.mmap)
                for line in lines:
                    next_line = next(wrapper)
                    assert next_line.strip() == line.strip()
                with pytest.raises(StopIteration, match='^$'):
                    next(wrapper)

    def test_unknown_engine(self) -> None:
        with tm.ensure_clean() as path:
            df = pd.DataFrame(1.1 * np.arange(120).reshape((30, 4)), columns=pd.Index(list('ABCD')), index=pd.Index([f'i-{i}' for i in range(30)]))
            df.to_csv(path)
            with pytest.raises(ValueError, match='Unknown engine'):
                pd.read_csv(path, engine='pyt')

    def test_binary_mode(self) -> None:
        """
        'encoding' shouldn't be passed to 'open' in binary mode.

        GH 35058
        """
        with tm.ensure_clean() as path:
            df = pd.DataFrame(1.1 * np.arange(120).reshape((30, 4)), columns=pd.Index(list('ABCD')), index=pd.Index([f'i-{i}' for i in range(30)]))
            df.to_csv(path, mode='w+b')
            tm.assert_frame_equal(df, pd.read_csv(path, index_col=0))

    @pytest.mark.parametrize('encoding', ['utf-16', 'utf-32'])
    @pytest.mark.parametrize('compression_', ['bz2', 'xz'])
    def test_warning_missing_utf_bom(self, encoding: Any, compression_: Any) -> None:
        """
        bz2 and xz do not write the byte order mark (BOM) for utf-16/32.

        https://stackoverflow.com/questions/55171439

        GH 35681
        """
        df = pd.DataFrame(1.1 * np.arange(120).reshape((30, 4)), columns=pd.Index(list('ABCD')), index=pd.Index([f'i-{i}' for i in range(30)]))
        with tm.ensure_clean() as path:
            with tm.assert_produces_warning(UnicodeWarning, match='byte order mark'):
                df.to_csv(path, compression=compression_, encoding=encoding)
            msg = "UTF-\\d+ stream does not start with BOM|'utf-\\d+' codec can't decode byte"
            with pytest.raises(UnicodeError, match=msg):
                pd.read_csv(path, compression=compression_, encoding=encoding)

def test_is_fsspec_url() -> None:
    assert icom.is_fsspec_url('gcs://pandas/somethingelse.com')
    assert icom.is_fsspec_url('gs://pandas/somethingelse.com')
    assert not icom.is_fsspec_url('http://pandas/somethingelse.com')
    assert not icom.is_fsspec_url('random:pandas/somethingelse.com')
    assert not icom.is_fsspec_url('/local/path')
    assert not icom.is_fsspec_url('relative/local/path')
    assert not icom.is_fsspec_url('this is not fsspec://url')
    assert not icom.is_fsspec_url("{'url': 'gs://pandas/somethingelse.com'}")
    assert icom.is_fsspec_url('RFC-3986+compliant.spec://something')

@pytest.mark.parametrize('encoding', [None, 'utf-8'])
@pytest.mark.parametrize('format', ['csv', 'json'])
def test_codecs_encoding(encoding: Any, format: Any) -> None:
    expected = pd.DataFrame(1.1 * np.arange(120).reshape((30, 4)), columns=pd.Index(list('ABCD')), index=pd.Index([f'i-{i}' for i in range(30)]))
    with tm.ensure_clean() as path:
        with codecs.open(path, mode='w', encoding=encoding) as handle:
            getattr(expected, f'to_{format}')(handle)
        with codecs.open(path, mode='r', encoding=encoding) as handle:
            if format == 'csv':
                df = pd.read_csv(handle, index_col=0)
            else:
                df = pd.read_json(handle)
    tm.assert_frame_equal(expected, df)

def test_codecs_get_writer_reader() -> None:
    expected = pd.DataFrame(1.1 * np.arange(120).reshape((30, 4)), columns=pd.Index(list('ABCD')), index=pd.Index([f'i-{i}' for i in range(30)]))
    with tm.ensure_clean() as path:
        with open(path, 'wb') as handle:
            with codecs.getwriter('utf-8')(handle) as encoded:
                expected.to_csv(encoded)
        with open(path, 'rb') as handle:
            with codecs.getreader('utf-8')(handle) as encoded:
                df = pd.read_csv(encoded, index_col=0)
    tm.assert_frame_equal(expected, df)

@pytest.mark.parametrize('io_class,mode,msg', [(BytesIO, 't', "a bytes-like object is required, not 'str'"), (StringIO, 'b', "string argument expected, got 'bytes'")])
def test_explicit_encoding(io_class: Any, mode: Any, msg: Any) -> None:
    expected = pd.DataFrame(1.1 * np.arange(120).reshape((30, 4)), columns=pd.Index(list('ABCD')), index=pd.Index([f'i-{i}' for i in range(30)]))
    with io_class() as buffer:
        with pytest.raises(TypeError, match=msg):
            expected.to_csv(buffer, mode=f'w{mode}')

@pytest.mark.parametrize('encoding_errors', ['strict', 'replace'])
@pytest.mark.parametrize('format', ['csv', 'json'])
def test_encoding_errors(encoding_errors: Any, format: Any) -> None:
    msg = "'utf-8' codec can't decode byte"
    bad_encoding = b'\xe4'
    if format == 'csv':
        content = b',' + bad_encoding + b'\n' + bad_encoding * 2 + b',' + bad_encoding
        reader = partial(pd.read_csv, index_col=0)
    else:
        content = b'{"' + bad_encoding * 2 + b'": {"' + bad_encoding + b'":"' + bad_encoding + b'"}}'
        reader = partial(pd.read_json, orient='index')
    with tm.ensure_clean() as path:
        file = Path(path)
        file.write_bytes(content)
        if encoding_errors != 'replace':
            with pytest.raises(UnicodeDecodeError, match=msg):
                reader(path, encoding_errors=encoding_errors)
        else:
            df = reader(path, encoding_errors=encoding_errors)
            decoded = bad_encoding.decode(errors=encoding_errors)
            expected = pd.DataFrame({decoded: [decoded]}, index=[decoded * 2])
            tm.assert_frame_equal(df, expected)

@pytest.mark.parametrize('encoding_errors', [0, None])
def test_encoding_errors_badtype(encoding_errors: Any) -> None:
    content = StringIO('A,B\n1,2\n3,4\n')
    reader = partial(pd.read_csv, encoding_errors=encoding_errors)
    expected_error = 'encoding_errors must be a string, got '
    expected_error += f'{type(encoding_errors).__name__}'
    with pytest.raises(ValueError, match=expected_error):
        reader(content)

def test_bad_encdoing_errors() -> None:
    with tm.ensure_clean() as path:
        with pytest.raises(LookupError, match='unknown error handler name'):
            icom.get_handle(path, 'w', errors='bad')

@pytest.mark.skipif(WASM, reason='limited file system access on WASM')
def test_errno_attribute() -> None:
    with pytest.raises(FileNotFoundError, match='\\[Errno 2\\]') as err:
        pd.read_csv('doesnt_exist')
        assert err.errno == errno.ENOENT

def test_fail_mmap() -> None:
    with pytest.raises(UnsupportedOperation, match='fileno'):
        with BytesIO() as buffer:
            icom.get_handle(buffer, 'rb', memory_map=True)

def test_close_on_error() -> None:

    class TestError:

        def close(self) -> None:
            raise OSError('test')
    with pytest.raises(OSError, match='test'):
        with BytesIO() as buffer:
            with icom.get_handle(buffer, 'rb') as handles:
                handles.created_handles.append(TestError())

@td.skip_if_no('fsspec', min_version='2023.1.0')
@pytest.mark.parametrize('compression', [None, 'infer'])
def test_read_csv_chained_url_no_error(compression: Any) -> None:
    tar_file_path = 'pandas/tests/io/data/tar/test-csv.tar'
    chained_file_url = f'tar://test.csv::file://{tar_file_path}'
    result = pd.read_csv(chained_file_url, compression=compression, sep=';')
    expected = pd.DataFrame({'1': {0: 3}, '2': {0: 4}})
    tm.assert_frame_equal(expected, result)

@pytest.mark.parametrize('reader', [pd.read_csv, pd.read_fwf, pd.read_excel, pd.read_feather, pd.read_hdf, pd.read_stata, pd.read_sas, pd.read_json, pd.read_pickle])
def test_pickle_reader(reader: Any) -> None:
    with BytesIO() as buffer:
        pickle.dump(reader, buffer)

@td.skip_if_no('pyarrow')
def test_pyarrow_read_csv_datetime_dtype() -> None:
    data = '"date"\n"20/12/2025"\n""\n"31/12/2020"'
    result = pd.read_csv(StringIO(data), parse_dates=['date'], dayfirst=True, dtype_backend='pyarrow')
    expect_data = pd.to_datetime(['20/12/2025', pd.NaT, '31/12/2020'], dayfirst=True)
    expect = pd.DataFrame({'date': expect_data})
    tm.assert_frame_equal(expect, result)