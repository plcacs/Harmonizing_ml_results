from __future__ import annotations
import logging
import pprint
import shutil
from decimal import Decimal
from fractions import Fraction
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Optional, Tuple, Union, List, Set, TypeVar, Type, cast
import fsspec
import pandas as pd
import pytest
from kedro.io.core import AbstractDataset, AbstractVersionedDataset, DatasetError, Version, VersionNotFoundError, generate_timestamp, get_filepath_str, get_protocol_and_path, parse_dataset_definition, validate_on_forbidden_chars
from kedro.io.lambda_dataset import LambdaDataset

T = TypeVar('T')
FALSE_BUILTINS: List[Any] = [False, 0, 0.0, 0.0j, Decimal(0), Fraction(0, 1), '', (), [], {}, set(), range(0)]


class MyDataset(AbstractDataset):

    def __init__(self, filepath: str = '', save_args: Optional[Dict[str, Any]] = None, fs_args: Optional[Dict[str, Any]] = None, var: Any = None) -> None:
        self._filepath: PurePosixPath = PurePosixPath(filepath)
        self.save_args: Optional[Dict[str, Any]] = save_args
        self.fs_args: Optional[Dict[str, Any]] = fs_args
        self.var: Any = var

    def func_hz1c16tz(self) -> Dict[str, Any]:
        return {'filepath': self._filepath, 'var': self.var}

    def func_tyquyq7u(self) -> bool:
        return Path(self._filepath.as_posix()).exists()

    def func_iwrmg7ca(self) -> pd.DataFrame:
        return pd.read_csv(self._filepath)

    def func_0k7zase9(self, data: str) -> None:
        with open(self._filepath, mode='w') as file:
            file.write(data)


class MyVersionedDataset(AbstractVersionedDataset[str, str]):

    def __init__(self, filepath: str, version: Optional[Version] = None) -> None:
        _fs_args: Dict[str, Any] = {}
        _fs_args.setdefault('auto_mkdir', True)
        protocol, path = get_protocol_and_path(filepath, version)
        self._protocol: str = protocol
        self._fs: fsspec.AbstractFileSystem = fsspec.filesystem(self._protocol, **_fs_args)
        super().__init__(filepath=PurePosixPath(path), version=version,
            exists_function=self._fs.exists, glob_function=self._fs.glob)

    def func_hz1c16tz(self) -> Dict[str, Any]:
        return dict(filepath=self._filepath, version=self._version)

    def func_iwrmg7ca(self) -> str:
        load_path = get_filepath_str(self._get_load_path(), self._protocol)
        with self._fs.open(load_path, mode='r') as fs_file:
            return fs_file.read()

    def func_0k7zase9(self, data: str) -> None:
        save_path = get_filepath_str(self._get_save_path(), self._protocol)
        with self._fs.open(save_path, mode='w') as fs_file:
            fs_file.write(data)

    def func_tyquyq7u(self) -> bool:
        try:
            load_path = get_filepath_str(self._get_load_path(), self._protocol)
        except DatasetError:
            return False
        return self._fs.exists(load_path)


class MyLocalVersionedDataset(AbstractVersionedDataset[str, str]):

    def __init__(self, filepath: str, version: Optional[Version] = None) -> None:
        _fs_args: Dict[str, Any] = {}
        _fs_args.setdefault('auto_mkdir', True)
        protocol, path = get_protocol_and_path(filepath, version)
        self._protocol: str = protocol
        self._fs: fsspec.AbstractFileSystem = fsspec.filesystem(self._protocol, **_fs_args)
        super().__init__(filepath=PurePosixPath(path), version=version,
            glob_function=self._fs.glob)

    def func_hz1c16tz(self) -> Dict[str, Any]:
        return dict(filepath=self._filepath, version=self._version)

    def func_iwrmg7ca(self) -> str:
        load_path = get_filepath_str(self._get_load_path(), self._protocol)
        with self._fs.open(load_path, mode='r') as fs_file:
            return fs_file.read()

    def func_0k7zase9(self, data: str) -> None:
        save_path = get_filepath_str(self._get_save_path(), self._protocol)
        with self._fs.open(save_path, mode='w') as fs_file:
            fs_file.write(data)

    def func_tyquyq7u(self) -> bool:
        load_path = get_filepath_str(self._get_load_path(), self._protocol)
        return self._fs.exists(load_path)


class MyOtherVersionedDataset(MyLocalVersionedDataset):

    def func_tyquyq7u(self) -> bool:
        try:
            load_path = get_filepath_str(self._get_load_path(), self._protocol)
        except VersionNotFoundError:
            raise NameError('Raising a NameError instead')
        return self._fs.exists(load_path)


@pytest.fixture(params=[None])
def func_s88k0kak(request: pytest.FixtureRequest) -> Any:
    return request.param


@pytest.fixture(params=[None])
def func_55txibfp(request: pytest.FixtureRequest) -> str:
    return request.param or generate_timestamp()


@pytest.fixture(params=[None])
def func_kfkc9p55(request: pytest.FixtureRequest) -> Any:
    return request.param


@pytest.fixture(params=[None])
def func_ldxfjmvy(request: pytest.FixtureRequest) -> Any:
    return request.param


@pytest.fixture(params=[None])
def func_1ln71an3(request: pytest.FixtureRequest) -> Any:
    return request.param


@pytest.fixture
def func_6ocyki9w(tmp_path: Path) -> str:
    return (tmp_path / 'test.csv').as_posix()


@pytest.fixture
def func_kja6kma0(filepath_versioned: str, save_args: Optional[Dict[str, Any]], fs_args: Optional[Dict[str, Any]]) -> MyDataset:
    return MyDataset(filepath=filepath_versioned, save_args=save_args,
        fs_args=fs_args)


@pytest.fixture
def func_i5pjz4k7(filepath_versioned: str, load_version: Optional[str], save_version: Optional[str]) -> MyVersionedDataset:
    return MyVersionedDataset(filepath=filepath_versioned, version=Version(
        load_version, save_version))


@pytest.fixture
def func_aggwf2hq() -> str:
    return 'col1 : [1, 2], col2 : [4, 5], col3 : [5, 6]}'


class TestCoreFunctions:

    @pytest.mark.parametrize('var', [1, True, *FALSE_BUILTINS])
    def func_su5rd94q(self, var: Any) -> None:
        var_str = pprint.pformat(var)
        filepath_str = pprint.pformat(PurePosixPath('.'))
        assert str(MyDataset(var=var)) == f'MyDataset(filepath=., var={var})'
        assert repr(MyDataset(var=var)
            ) == f'tests.io.test_core.MyDataset(filepath={filepath_str}, var={var_str})'

    def func_ochrbcqk(self) -> None:
        assert str(MyDataset()) == 'MyDataset(filepath=.)'
        filepath_str = pprint.pformat(PurePosixPath('.'))
        assert repr(MyDataset()
            ) == f'tests.io.test_core.MyDataset(filepath={filepath_str})'

    @pytest.mark.parametrize('describe_return', [None, {'key_1': 'val_1', (
        2): 'val_2'}])
    def func_ijv3ic4d(self, describe_return: Optional[Dict[Union[str, int], str]], caplog: pytest.LogCaptureFixture) -> None:


        class BadDescribeDataset(MyDataset):

            def func_hz1c16tz(self) -> Optional[Dict[Union[str, int], str]]:
                return describe_return
        warning_message = (
            "'tests.io.test_core.BadDescribeDataset' is a subclass of AbstractDataset and it must implement the '_describe' method following the signature of AbstractDataset's '_describe'."
            )
        with caplog.at_level(logging.WARNING):
            assert repr(BadDescribeDataset()
                ) == 'tests.io.test_core.BadDescribeDataset()'
            assert warning_message in caplog.text

    def func_z00nuect(self) -> None:
        path = get_filepath_str(PurePosixPath('example.com/test.csv'), 'http')
        assert isinstance(path, str)
        assert path == 'http://example.com/test.csv'

    @pytest.mark.parametrize('filepath,expected_result', [(
        's3://bucket/file.txt', ('s3', 'bucket/file.txt')), (
        's3://user@BUCKET/file.txt', ('s3', 'BUCKET/file.txt')), (
        'gcs://bucket/file.txt', ('gcs', 'bucket/file.txt')), (
        'gs://bucket/file.txt', ('gs', 'bucket/file.txt')), (
        'adl://bucket/file.txt', ('adl', 'bucket/file.txt')), (
        'abfs://bucket/file.txt', ('abfs', 'bucket/file.txt')), (
        'abfss://bucket/file.txt', ('abfss', 'bucket/file.txt')), (
        'abfss://mycontainer@mystorageaccount.dfs.core.windows.net/mypath',
        ('abfss',
        'mycontainer@mystorageaccount.dfs.core.windows.net/mypath')), (
        'oci://bucket@namespace/file.txt', ('oci',
        'bucket@namespace/file.txt')), ('hdfs://namenode:8020/file.txt', (
        'hdfs', '/file.txt')), ('file:///tmp/file.txt', ('file',
        '/tmp/file.txt')), ('/tmp/file.txt', ('file', '/tmp/file.txt')), (
        'C:\\Projects\\file.txt', ('file', 'C:\\Projects\\file.txt')), (
        'file:///C:\\Projects\\file.txt', ('file', 'C:\\Projects\\file.txt'
        )), ('https://example.com/file.txt', ('https',
        'example.com/file.txt')), ('http://example.com/file.txt', ('http',
        'example.com/file.txt')), (
        'https://example.com/search?query=books&category=fiction#reviews',
        ('https', 'example.com/search?query=books&category=fiction#reviews'
        )), ('https://example.com/search#reviews', ('https',
        'example.com/search#reviews')), (
        'http://example.com/search?query=books&category=fiction', ('http',
        'example.com/search?query=books&category=fiction')), (
        's3://some/example?query=query#filename', ('s3',
        'some/example?query=query#filename')), (
        's3://some/example#filename', ('s3', 'some/example#filename')), (
        's3://some/example?query=query', ('s3', 'some/example?query=query'))])
    def func_aac7dyb6(self, filepath: str, expected_result: Tuple[str, str]) -> None:
        assert get_protocol_and_path(filepath) == expected_result

    @pytest.mark.parametrize('filepath', ['http://example.com/file.txt',
        'https://example.com/file.txt'])
    def func_n6o8tcdf(self, filepath: str) -> None:
        version = Version(load=None, save=None)
        expected_error_message = (
            'Versioning is not supported for HTTP protocols. Please remove the `versioned` flag from the dataset configuration.'
            )
        with pytest.raises(DatasetError, match=expected_error_message):
            get_protocol_and_path(filepath, version)

    @pytest.mark.parametrize('input', [{'key1': 'invalid value'}, {'key2':
        'invalid;value'}])
    def func_4q9czf93(self, input: Dict[str, str]) -> None:
        key = next(iter(input.keys()))
        expected_error_message = (
            f"Neither white-space nor semicolon are allowed in '{key}'.")
        with pytest.raises(DatasetError, match=expected_error_message):
            validate_on_forbidden_chars(**input)

    def func_721d0qoy(self, mocker: pytest.MockFixture) -> None:
        mocker.patch('kedro.io.core.load_obj', return_value=None)
        dataset_name = 'lAmbDaDaTAsET'
        with pytest.raises(DatasetError, match=
            f"Class '{dataset_name}' not found, is this a typo?"):
            parse_dataset_definition({'type': dataset_name})

    def func_ol4lhust(self, mocker: pytest.MockFixture) -> None:
        dataset_name = 'LambdaDataset'

        def func_1efo2na1(value: str) -> List[str]:
            if '__all__' in value:
                return [dataset_name]
            else:
                raise ModuleNotFoundError
        mocker.patch('kedro.io.core.load_obj', side_effect=func_1efo2na1)
        pattern = (
            'Please see the documentation on how to install relevant dependencies'
            )
        with pytest.raises(DatasetError, match=pattern):
            parse_dataset_definition({'type': dataset_name})

    def func_upta9ed2(self) -> None:
        config = {'type': 'LambdaDataset'}
        dataset, _ = parse_dataset_definition(config)
        assert dataset is LambdaDataset

    def func_u9obu0g5(self) -> None:
        config = {'type': MyDataset}
        parse_dataset_definition(config)

    def func_1iqy9vaj(self) -> None:
        assert not getattr(MyOtherVersionedDataset.load.__wrapped__,
            '__loadwrapped__', False)
        assert not getattr(MyOtherVersionedDataset.save.__wrapped__,
            '__savewrapped__', False)


class TestAbstractVersionedDataset:

    def func_biallnba(self, load_version: Optional[str], save_version: Optional[str]) -> None:
        """Test that version is in string representation of the class instance
        when applicable."""
        filepath = 'test.csv'
        ds_versioned = MyVersionedDataset(filepath=filepath, version=
            Version(load_version, save_version))
        assert filepath in str(ds_versioned)
        ver_str = (
            f"version=Version(load={load_version}, save='{save_version}')")
        assert ver_str in str(ds_versioned)
        assert 'MyVersionedDataset' in str(ds_versioned)

    def func_feubw5wv(self, my_versioned_dataset: MyVersionedDataset, dummy_data: str) -> None:
        """Test that saved and reloaded data matches the original one for
        the versioned dataset."""
        func_i5pjz4k7.save(dummy_data)
        reloaded = func_i5pjz4k7.load()
        assert dummy_data == reloaded

    def func_qh34l4ol(self, dummy_data: str) -> None:
        ds = MyVersionedDataset('test.csv', Version(None, None))
        ds.save(dummy_data)
        assert ds._filepath
        shutil.rmtree(ds._filepath)

    def func_8623pzzf(self, my_versioned_dataset: MyVersionedDataset) -> None:
        """Check the error if no versions are available for load."""
        pattern = 'Did not find any versions for MyVersionedDataset\\(.+\\)'
        with pytest.raises(DatasetError, match=pattern):
            func_i5pjz4k7.load()

    def func_lvimhk6i(self, dummy_data: str) -> None:
        """Check the error if no versions are available for load (_local_exists())."""
        version = Version(load=None, save=None)
        my_versioned_dataset = MyLocalVersionedDataset('test.csv', version=
            version)
        assert func_i5pjz4k7.exists() is False
        func_i5pjz4k7.save(dummy_data)
        assert func_i5pjz4k7.exists() is True
        shutil.rmtree(my_versioned_dataset._filepath)

    def func_hkvvj6s7(self) -> None:
        """Check if all exceptions are shown as DatasetError for exists() check"""
        version = Version(load=None, save=None)
        my_other_versioned_dataset = MyOtherVersionedDataset('test.csv',
            version=version)
        with pytest.raises(DatasetError):
            my_other_versioned_dataset.exists()

    def func_sg0bh332(self, my_versioned_dataset: MyVersionedDataset, dummy_data: str) -> None:
        """Test `exists` method invocation for versioned dataset."""
        assert not func_i5pjz4k7.exists()
        func_i5pjz4k7.save(dummy_data)
        assert func_i5pjz4k7.exists()
        shutil.rmtree(my_versioned_dataset._filepath)

    def func_vgmkkz7z(self, my_versioned_dataset: MyVersionedDataset, dummy_data: str) -> None:
        """Check the error when attempting to override the dataset if the
        corresponding json file for a given save version already exists."""
        func_i5pjz4k7.save(dummy_data)
        pattern = (
            "Save path \\'.+\\' for MyVersionedDataset\\(.+\\) must not exist if versioning is enabled\\."
            )
        with pytest.raises(DatasetError, match=pattern):
            func_i5pjz4k7.save(dummy_data)

    @pytest.mark.parametrize('load_version', ['2019-01-01T23