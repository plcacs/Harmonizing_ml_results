from __future__ import annotations
import logging
import pprint
import shutil
from decimal import Decimal
from fractions import Fraction
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Optional, Tuple, Union

import fsspec
import pandas as pd
import pytest
from kedro.io.core import (
    AbstractDataset,
    AbstractVersionedDataset,
    DatasetError,
    Version,
    VersionNotFoundError,
    generate_timestamp,
    get_filepath_str,
    get_protocol_and_path,
    parse_dataset_definition,
    validate_on_forbidden_chars,
)
from kedro.io.lambda_dataset import LambdaDataset

FALSE_BUILTINS: list[Any] = [
    False,
    0,
    0.0,
    0j,
    Decimal(0),
    Fraction(0, 1),
    "",
    (),
    [],
    {},
    set(),
    range(0),
]


class MyDataset(AbstractDataset):
    def __init__(
        self,
        filepath: str = "",
        save_args: Optional[Any] = None,
        fs_args: Optional[Any] = None,
        var: Optional[Any] = None,
    ) -> None:
        self._filepath: PurePosixPath = PurePosixPath(filepath or ".")
        self.save_args = save_args
        self.fs_args = fs_args
        self.var = var

    def _describe(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"filepath": self._filepath}
        if self.var is not None:
            d["var"] = self.var
        return d

    def _exists(self) -> bool:
        return Path(self._filepath.as_posix()).exists()

    def _load(self) -> pd.DataFrame:
        return pd.read_csv(self._filepath.as_posix())

    def _save(self, data: str) -> None:
        with open(self._filepath.as_posix(), mode="w") as file:
            file.write(data)


class MyVersionedDataset(AbstractVersionedDataset[str, str]):
    def __init__(self, filepath: str, version: Optional[Version] = None) -> None:
        _fs_args: Dict[str, Any] = {}
        _fs_args.setdefault("auto_mkdir", True)
        protocol, path = get_protocol_and_path(filepath, version)
        self._protocol: str = protocol
        self._fs = fsspec.filesystem(self._protocol, **_fs_args)
        super().__init__(
            filepath=PurePosixPath(path),
            version=version,
            exists_function=self._fs.exists,
            glob_function=self._fs.glob,
        )

    def _describe(self) -> Dict[str, Any]:
        return dict(filepath=self._filepath, version=self._version)

    def _load(self) -> str:
        load_path: str = get_filepath_str(self._get_load_path(), self._protocol)
        with self._fs.open(load_path, mode="r") as fs_file:
            return fs_file.read()

    def _save(self, data: str) -> None:
        save_path: str = get_filepath_str(self._get_save_path(), self._protocol)
        with self._fs.open(save_path, mode="w") as fs_file:
            fs_file.write(data)

    def _exists(self) -> bool:
        try:
            load_path: str = get_filepath_str(self._get_load_path(), self._protocol)
        except DatasetError:
            return False
        return self._fs.exists(load_path)


class MyLocalVersionedDataset(AbstractVersionedDataset[str, str]):
    def __init__(self, filepath: str, version: Optional[Version] = None) -> None:
        _fs_args: Dict[str, Any] = {}
        _fs_args.setdefault("auto_mkdir", True)
        protocol, path = get_protocol_and_path(filepath, version)
        self._protocol: str = protocol
        self._fs = fsspec.filesystem(self._protocol, **_fs_args)
        super().__init__(
            filepath=PurePosixPath(path),
            version=version,
            glob_function=self._fs.glob,
        )

    def _describe(self) -> Dict[str, Any]:
        return dict(filepath=self._filepath, version=self._version)

    def _load(self) -> str:
        load_path: str = get_filepath_str(self._get_load_path(), self._protocol)
        with self._fs.open(load_path, mode="r") as fs_file:
            return fs_file.read()

    def _save(self, data: str) -> None:
        save_path: str = get_filepath_str(self._get_save_path(), self._protocol)
        with self._fs.open(save_path, mode="w") as fs_file:
            fs_file.write(data)

    def _exists(self) -> bool:
        load_path: str = get_filepath_str(self._get_load_path(), self._protocol)
        return self._fs.exists(load_path)


class MyOtherVersionedDataset(MyLocalVersionedDataset):
    def _exists(self) -> bool:
        try:
            load_path: str = get_filepath_str(self._get_load_path(), self._protocol)
        except VersionNotFoundError:
            raise NameError("Raising a NameError instead")
        return self._fs.exists(load_path)


@pytest.fixture(params=[None])
def load_version(request: Any) -> Any:
    return request.param


@pytest.fixture(params=[None])
def save_version(request: Any) -> Any:
    return request.param or generate_timestamp()


@pytest.fixture(params=[None])
def load_args(request: Any) -> Any:
    return request.param


@pytest.fixture(params=[None])
def save_args(request: Any) -> Any:
    return request.param


@pytest.fixture(params=[None])
def fs_args(request: Any) -> Any:
    return request.param


@pytest.fixture
def filepath_versioned(tmp_path: Path) -> str:
    return (tmp_path / "test.csv").as_posix()


@pytest.fixture
def my_dataset(filepath_versioned: str, save_args: Any, fs_args: Any) -> MyDataset:
    return MyDataset(filepath=filepath_versioned, save_args=save_args, fs_args=fs_args)


@pytest.fixture
def my_versioned_dataset(
    filepath_versioned: str, load_version: Any, save_version: Any
) -> MyVersionedDataset:
    return MyVersionedDataset(filepath=filepath_versioned, version=Version(load_version, save_version))


@pytest.fixture
def dummy_data() -> str:
    return "col1 : [1, 2], col2 : [4, 5], col3 : [5, 6]}"


class TestCoreFunctions:
    @pytest.mark.parametrize("var", [1, True, *FALSE_BUILTINS])
    def test_str_representation(self, var: Any) -> None:
        var_str: str = pprint.pformat(var)
        filepath_str: str = pprint.pformat(PurePosixPath("."))
        assert str(MyDataset(var=var)) == f"MyDataset(filepath=., var={var})"
        assert repr(MyDataset(var=var)) == f"tests.io.test_core.MyDataset(filepath={filepath_str}, var={var_str})"

    def test_str_representation_none(self) -> None:
        assert str(MyDataset()) == "MyDataset(filepath=.)"
        filepath_str: str = pprint.pformat(PurePosixPath("."))
        assert repr(MyDataset()) == f"tests.io.test_core.MyDataset(filepath={filepath_str})"

    @pytest.mark.parametrize("describe_return", [None, {"key_1": "val_1", 2: "val_2"}])
    def test_repr_bad_describe(self, describe_return: Any, caplog: Any) -> None:
        class BadDescribeDataset(MyDataset):
            def _describe(self) -> Any:
                return describe_return

        warning_message = (
            "'tests.io.test_core.BadDescribeDataset' is a subclass of AbstractDataset "
            "and it must implement the '_describe' method following the signature of AbstractDataset's '_describe'."
        )
        with caplog.at_level(logging.WARNING):
            assert repr(BadDescribeDataset()) == "tests.io.test_core.BadDescribeDataset()"
            assert warning_message in caplog.text

    def test_get_filepath_str(self) -> None:
        path: PurePosixPath = PurePosixPath("example.com/test.csv")
        result: str = get_filepath_str(path, "http")
        assert isinstance(result, str)
        assert result == "http://example.com/test.csv"

    @pytest.mark.parametrize(
        "filepath,expected_result",
        [
            ("s3://bucket/file.txt", ("s3", "bucket/file.txt")),
            ("s3://user@BUCKET/file.txt", ("s3", "BUCKET/file.txt")),
            ("gcs://bucket/file.txt", ("gcs", "bucket/file.txt")),
            ("gs://bucket/file.txt", ("gs", "bucket/file.txt")),
            ("adl://bucket/file.txt", ("adl", "bucket/file.txt")),
            ("abfs://bucket/file.txt", ("abfs", "bucket/file.txt")),
            ("abfss://bucket/file.txt", ("abfss", "bucket/file.txt")),
            (
                "abfss://mycontainer@mystorageaccount.dfs.core.windows.net/mypath",
                ("abfss", "mycontainer@mystorageaccount.dfs.core.windows.net/mypath"),
            ),
            ("oci://bucket@namespace/file.txt", ("oci", "bucket@namespace/file.txt")),
            ("hdfs://namenode:8020/file.txt", ("hdfs", "/file.txt")),
            ("file:///tmp/file.txt", ("file", "/tmp/file.txt")),
            ("/tmp/file.txt", ("file", "/tmp/file.txt")),
            ("C:\\Projects\\file.txt", ("file", "C:\\Projects\\file.txt")),
            ("file:///C:\\Projects\\file.txt", ("file", "C:\\Projects\\file.txt")),
            ("https://example.com/file.txt", ("https", "example.com/file.txt")),
            ("http://example.com/file.txt", ("http", "example.com/file.txt")),
            (
                "https://example.com/search?query=books&category=fiction#reviews",
                ("https", "example.com/search?query=books&category=fiction#reviews"),
            ),
            ("https://example.com/search#reviews", ("https", "example.com/search#reviews")),
            (
                "http://example.com/search?query=books&category=fiction",
                ("http", "example.com/search?query=books&category=fiction"),
            ),
            (
                "s3://some/example?query=query#filename",
                ("s3", "some/example?query=query#filename"),
            ),
            ("s3://some/example#filename", ("s3", "some/example#filename")),
            ("s3://some/example?query=query", ("s3", "some/example?query=query")),
        ],
    )
    def test_get_protocol_and_path(self, filepath: str, expected_result: Tuple[str, str]) -> None:
        assert get_protocol_and_path(filepath) == expected_result

    @pytest.mark.parametrize("filepath", ["http://example.com/file.txt", "https://example.com/file.txt"])
    def test_get_protocol_and_path_http_with_version(self, filepath: str) -> None:
        version: Version = Version(load=None, save=None)
        expected_error_message = (
            "Versioning is not supported for HTTP protocols. Please remove the `versioned` flag from the dataset configuration."
        )
        with pytest.raises(DatasetError, match=expected_error_message):
            get_protocol_and_path(filepath, version)

    @pytest.mark.parametrize("input", [{"key1": "invalid value"}, {"key2": "invalid;value"}])
    def test_validate_forbidden_chars(self, input: Dict[str, str]) -> None:
        key = next(iter(input.keys()))
        expected_error_message = f"Neither white-space nor semicolon are allowed in '{key}'."
        with pytest.raises(DatasetError, match=expected_error_message):
            validate_on_forbidden_chars(**input)

    def test_dataset_name_typo(self, mocker: Any) -> None:
        mocker.patch("kedro.io.core.load_obj", return_value=None)
        dataset_name: str = "lAmbDaDaTAsET"
        with pytest.raises(DatasetError, match=f"Class '{dataset_name}' not found, is this a typo?"):
            parse_dataset_definition({"type": dataset_name})

    def test_dataset_missing_dependencies(self, mocker: Any) -> None:
        dataset_name: str = "LambdaDataset"

        def side_effect_function(value: Any) -> Any:
            if "__all__" in value:
                return [dataset_name]
            else:
                raise ModuleNotFoundError

        mocker.patch("kedro.io.core.load_obj", side_effect=side_effect_function)
        pattern: str = "Please see the documentation on how to install relevant dependencies"
        with pytest.raises(DatasetError, match=pattern):
            parse_dataset_definition({"type": dataset_name})

    def test_parse_dataset_definition(self) -> None:
        config: Dict[str, Any] = {"type": "LambdaDataset"}
        dataset, _ = parse_dataset_definition(config)
        assert dataset is LambdaDataset

    def test_parse_dataset_definition_with_python_class_type(self) -> None:
        config: Dict[str, Any] = {"type": MyDataset}
        parse_dataset_definition(config)

    def test_load_and_save_are_wrapped_once(self) -> None:
        assert not getattr(MyOtherVersionedDataset.load.__wrapped__, "__loadwrapped__", False)
        assert not getattr(MyOtherVersionedDataset.save.__wrapped__, "__savewrapped__", False)


class TestAbstractVersionedDataset:
    def test_version_str_repr(self, load_version: Any, save_version: Any) -> None:
        filepath: str = "test.csv"
        ds_versioned = MyVersionedDataset(filepath=filepath, version=Version(load_version, save_version))
        assert filepath in str(ds_versioned)
        ver_str: str = f"version=Version(load={load_version}, save='{save_version}')"
        assert ver_str in str(ds_versioned)
        assert "MyVersionedDataset" in str(ds_versioned)

    def test_save_and_load(self, my_versioned_dataset: MyVersionedDataset, dummy_data: str) -> None:
        my_versioned_dataset.save(dummy_data)
        reloaded: str = my_versioned_dataset.load()
        assert dummy_data == reloaded

    def test_resolve_save_version(self, dummy_data: str) -> None:
        ds = MyVersionedDataset("test.csv", Version(None, None))
        ds.save(dummy_data)
        assert ds._filepath
        shutil.rmtree(ds._filepath)

    def test_no_versions(self, my_versioned_dataset: MyVersionedDataset) -> None:
        pattern = "Did not find any versions for MyVersionedDataset\\(.+\\)"
        with pytest.raises(DatasetError, match=pattern):
            my_versioned_dataset.load()

    def test_local_exists(self, dummy_data: str) -> None:
        version = Version(load=None, save=None)
        my_versioned_dataset = MyLocalVersionedDataset("test.csv", version=version)
        assert my_versioned_dataset.exists() is False
        my_versioned_dataset.save(dummy_data)
        assert my_versioned_dataset.exists() is True
        shutil.rmtree(my_versioned_dataset._filepath)

    def test_exists_general_exception(self) -> None:
        version = Version(load=None, save=None)
        my_other_versioned_dataset = MyOtherVersionedDataset("test.csv", version=version)
        with pytest.raises(DatasetError):
            my_other_versioned_dataset.exists()

    def test_exists(self, my_versioned_dataset: MyVersionedDataset, dummy_data: str) -> None:
        assert not my_versioned_dataset.exists()
        my_versioned_dataset.save(dummy_data)
        assert my_versioned_dataset.exists()
        shutil.rmtree(my_versioned_dataset._filepath)

    def test_prevent_overwrite(self, my_versioned_dataset: MyVersionedDataset, dummy_data: str) -> None:
        my_versioned_dataset.save(dummy_data)
        pattern = (
            "Save path \'.+\' for MyVersionedDataset\\(.+\\) must not exist if versioning is enabled\\."
        )
        with pytest.raises(DatasetError, match=pattern):
            my_versioned_dataset.save(dummy_data)

    @pytest.mark.parametrize("load_version", ["2019-01-01T23.59.59.999Z"], indirect=True)
    @pytest.mark.parametrize("save_version", ["2019-01-02T00.00.00.000Z"], indirect=True)
    def test_save_version_warning(
        self, my_versioned_dataset: MyVersionedDataset, load_version: Any, save_version: Any, dummy_data: str
    ) -> None:
        pattern = f"Save version '{save_version}' did not match load version '{load_version}' for MyVersionedDataset\\(.+\\)"
        with pytest.warns(UserWarning, match=pattern):
            my_versioned_dataset.save(dummy_data)

    def test_versioning_existing_dataset(
        self, my_dataset: MyDataset, my_versioned_dataset: MyVersionedDataset, dummy_data: str
    ) -> None:
        my_dataset.save(dummy_data)
        assert my_dataset.exists()
        assert my_dataset._filepath == my_versioned_dataset._filepath
        pattern = (
            f'(?=.*file with the same name already exists in the directory)(?=.*{my_versioned_dataset._filepath.parent.as_posix()})'
        )
        with pytest.raises(DatasetError, match=pattern):
            my_versioned_dataset.save(dummy_data)
        Path(my_dataset._filepath.as_posix()).unlink()
        my_versioned_dataset.save(dummy_data)
        assert my_versioned_dataset.exists()

    def test_cache_release(self, my_versioned_dataset: MyVersionedDataset) -> None:
        my_versioned_dataset._version_cache["index"] = "value"
        assert my_versioned_dataset._version_cache.currsize > 0
        my_versioned_dataset._release()
        assert my_versioned_dataset._version_cache.currsize == 0


class MyLegacyDataset(AbstractDataset):
    def __init__(
        self,
        filepath: str = "",
        save_args: Optional[Any] = None,
        fs_args: Optional[Any] = None,
        var: Optional[Any] = None,
    ) -> None:
        self._filepath: PurePosixPath = PurePosixPath(filepath or ".")
        self.save_args = save_args
        self.fs_args = fs_args
        self.var = var

    def _describe(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"filepath": self._filepath}
        if self.var is not None:
            d["var"] = self.var
        return d

    def _exists(self) -> bool:
        return Path(self._filepath.as_posix()).exists()

    def _load(self) -> pd.DataFrame:
        return pd.read_csv(self._filepath.as_posix())

    def _save(self, data: str) -> None:
        with open(self._filepath.as_posix(), mode="w") as file:
            file.write(data)


class MyLegacyVersionedDataset(AbstractVersionedDataset[str, str]):
    def __init__(self, filepath: str, version: Optional[Version] = None) -> None:
        _fs_args: Dict[str, Any] = {}
        _fs_args.setdefault("auto_mkdir", True)
        protocol, path = get_protocol_and_path(filepath, version)
        self._protocol: str = protocol
        self._fs = fsspec.filesystem(self._protocol, **_fs_args)
        super().__init__(
            filepath=PurePosixPath(path),
            version=version,
            exists_function=self._fs.exists,
            glob_function=self._fs.glob,
        )

    def _describe(self) -> Dict[str, Any]:
        return dict(filepath=self._filepath, version=self._version)

    def _load(self) -> str:
        load_path: str = get_filepath_str(self._get_load_path(), self._protocol)
        with self._fs.open(load_path, mode="r") as fs_file:
            return fs_file.read()

    def _save(self, data: str) -> None:
        save_path: str = get_filepath_str(self._get_save_path(), self._protocol)
        with self._fs.open(save_path, mode="w") as fs_file:
            fs_file.write(data)

    def _exists(self) -> bool:
        try:
            load_path: str = get_filepath_str(self._get_load_path(), self._protocol)
        except DatasetError:
            return False
        return self._fs.exists(load_path)


@pytest.fixture
def my_legacy_dataset(filepath_versioned: str, save_args: Any, fs_args: Any) -> MyLegacyDataset:
    return MyLegacyDataset(filepath=filepath_versioned, save_args=save_args, fs_args=fs_args)


@pytest.fixture
def my_legacy_versioned_dataset(
    filepath_versioned: str, load_version: Any, save_version: Any
) -> MyLegacyVersionedDataset:
    return MyLegacyVersionedDataset(filepath=filepath_versioned, version=Version(load_version, save_version))


class TestLegacyLoadAndSave:
    def test_saving_none(self, my_legacy_dataset: MyLegacyDataset) -> None:
        pattern = "Saving 'None' to a 'Dataset' is not allowed"
        with pytest.raises(DatasetError, match=pattern):
            my_legacy_dataset.save(None)  # type: ignore

    def test_saving_invalid_data(self, my_legacy_dataset: MyLegacyDataset, dummy_data: str) -> None:
        pattern = "Failed while saving data to dataset"
        with pytest.raises(DatasetError, match=pattern):
            my_legacy_dataset.save(pd.DataFrame())

    @pytest.mark.parametrize("load_version", ["2019-01-01T23.59.59.999Z"], indirect=True)
    @pytest.mark.parametrize("save_version", ["2019-01-02T00.00.00.000Z"], indirect=True)
    def test_save_version_warning(
        self, my_legacy_versioned_dataset: MyLegacyVersionedDataset, load_version: Any, save_version: Any, dummy_data: str
    ) -> None:
        pattern = f"Save version '{save_version}' did not match load version '{load_version}' for MyLegacyVersionedDataset\\(.+\\)"
        with pytest.warns(UserWarning, match=pattern):
            my_legacy_versioned_dataset.save(dummy_data)

    def test_versioning_existing_dataset(
        self, my_legacy_dataset: MyLegacyDataset, my_legacy_versioned_dataset: MyLegacyVersionedDataset, dummy_data: str
    ) -> None:
        my_legacy_dataset.save(dummy_data)
        assert my_legacy_dataset.exists()
        assert my_legacy_dataset._filepath == my_legacy_versioned_dataset._filepath
        pattern = (
            f'(?=.*file with the same name already exists in the directory)(?=.*{my_legacy_versioned_dataset._filepath.parent.as_posix()})'
        )
        with pytest.raises(DatasetError, match=pattern):
            my_legacy_versioned_dataset.save(dummy_data)
        Path(my_legacy_dataset._filepath.as_posix()).unlink()
        my_legacy_versioned_dataset.save(dummy_data)
        assert my_legacy_versioned_dataset.exists()