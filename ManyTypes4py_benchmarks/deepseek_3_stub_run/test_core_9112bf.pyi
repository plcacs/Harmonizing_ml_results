from __future__ import annotations
import logging
from pathlib import Path, PurePosixPath
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Union
from decimal import Decimal
from fractions import Fraction
import fsspec
import pandas as pd
import pytest
from kedro.io.core import (
    AbstractDataset,
    AbstractVersionedDataset,
    DatasetError,
    Version,
    VersionNotFoundError,
)

FALSE_BUILTINS: List[Any] = ...

class MyDataset(AbstractDataset):
    _filepath: PurePosixPath
    save_args: Optional[Any]
    fs_args: Optional[Any]
    var: Optional[Any]
    
    def __init__(
        self,
        filepath: str = "",
        save_args: Optional[Any] = None,
        fs_args: Optional[Any] = None,
        var: Optional[Any] = None,
    ) -> None: ...
    
    def _describe(self) -> Dict[str, Any]: ...
    def _exists(self) -> bool: ...
    def _load(self) -> pd.DataFrame: ...
    def _save(self, data: Any) -> None: ...

class MyVersionedDataset(AbstractVersionedDataset[str, str]):
    _protocol: str
    _fs: fsspec.AbstractFileSystem
    
    def __init__(self, filepath: str, version: Optional[Version] = None) -> None: ...
    def _describe(self) -> Dict[str, Any]: ...
    def _load(self) -> str: ...
    def _save(self, data: str) -> None: ...
    def _exists(self) -> bool: ...

class MyLocalVersionedDataset(AbstractVersionedDataset[str, str]):
    _protocol: str
    _fs: fsspec.AbstractFileSystem
    
    def __init__(self, filepath: str, version: Optional[Version] = None) -> None: ...
    def _describe(self) -> Dict[str, Any]: ...
    def _load(self) -> str: ...
    def _save(self, data: str) -> None: ...
    def _exists(self) -> bool: ...

class MyOtherVersionedDataset(MyLocalVersionedDataset):
    def _exists(self) -> bool: ...

def load_version(request: pytest.FixtureRequest) -> Optional[Any]: ...
def save_version(request: pytest.FixtureRequest) -> Optional[str]: ...
def load_args(request: pytest.FixtureRequest) -> Optional[Any]: ...
def save_args(request: pytest.FixtureRequest) -> Optional[Any]: ...
def fs_args(request: pytest.FixtureRequest) -> Optional[Any]: ...
def filepath_versioned(tmp_path: Path) -> str: ...
def my_dataset(
    filepath_versioned: str,
    save_args: Optional[Any],
    fs_args: Optional[Any],
) -> MyDataset: ...
def my_versioned_dataset(
    filepath_versioned: str,
    load_version: Optional[Any],
    save_version: Optional[str],
) -> MyVersionedDataset: ...
def dummy_data() -> str: ...

class TestCoreFunctions:
    @pytest.mark.parametrize("var", [1, True, *FALSE_BUILTINS])
    def test_str_representation(self, var: Any) -> None: ...
    def test_str_representation_none(self) -> None: ...
    @pytest.mark.parametrize("describe_return", [None, {"key_1": "val_1", 2: "val_2"}])
    def test_repr_bad_describe(
        self,
        describe_return: Optional[Dict[Union[str, int], str]],
        caplog: pytest.LogCaptureFixture,
    ) -> None: ...
    def test_get_filepath_str(self) -> None: ...
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
    def test_get_protocol_and_path(
        self, filepath: str, expected_result: Tuple[str, str]
    ) -> None: ...
    @pytest.mark.parametrize(
        "filepath", ["http://example.com/file.txt", "https://example.com/file.txt"]
    )
    def test_get_protocol_and_path_http_with_version(self, filepath: str) -> None: ...
    @pytest.mark.parametrize(
        "input", [{"key1": "invalid value"}, {"key2": "invalid;value"}]
    )
    def test_validate_forbidden_chars(self, input: Dict[str, str]) -> None: ...
    def test_dataset_name_typo(self, mocker: pytest.MockerFixture) -> None: ...
    def test_dataset_missing_dependencies(self, mocker: pytest.MockerFixture) -> None: ...
    def test_parse_dataset_definition(self) -> None: ...
    def test_parse_dataset_definition_with_python_class_type(self) -> None: ...
    def test_load_and_save_are_wrapped_once(self) -> None: ...

class TestAbstractVersionedDataset:
    def test_version_str_repr(
        self, load_version: Optional[Any], save_version: Optional[str]
    ) -> None: ...
    def test_save_and_load(
        self, my_versioned_dataset: MyVersionedDataset, dummy_data: str
    ) -> None: ...
    def test_resolve_save_version(self, dummy_data: str) -> None: ...
    def test_no_versions(self, my_versioned_dataset: MyVersionedDataset) -> None: ...
    def test_local_exists(self, dummy_data: str) -> None: ...
    def test_exists_general_exception(self) -> None: ...
    def test_exists(
        self, my_versioned_dataset: MyVersionedDataset, dummy_data: str
    ) -> None: ...
    def test_prevent_overwrite(
        self, my_versioned_dataset: MyVersionedDataset, dummy_data: str
    ) -> None: ...
    @pytest.mark.parametrize("load_version", ["2019-01-01T23.59.59.999Z"], indirect=True)
    @pytest.mark.parametrize("save_version", ["2019-01-02T00.00.00.000Z"], indirect=True)
    def test_save_version_warning(
        self,
        my_versioned_dataset: MyVersionedDataset,
        load_version: str,
        save_version: str,
        dummy_data: str,
    ) -> None: ...
    def test_versioning_existing_dataset(
        self,
        my_dataset: MyDataset,
        my_versioned_dataset: MyVersionedDataset,
        dummy_data: str,
    ) -> None: ...
    def test_cache_release(self, my_versioned_dataset: MyVersionedDataset) -> None: ...

class MyLegacyDataset(AbstractDataset):
    _filepath: PurePosixPath
    save_args: Optional[Any]
    fs_args: Optional[Any]
    var: Optional[Any]
    
    def __init__(
        self,
        filepath: str = "",
        save_args: Optional[Any] = None,
        fs_args: Optional[Any] = None,
        var: Optional[Any] = None,
    ) -> None: ...
    
    def _describe(self) -> Dict[str, Any]: ...
    def _exists(self) -> bool: ...
    def _load(self) -> pd.DataFrame: ...
    def _save(self, data: Any) -> None: ...

class MyLegacyVersionedDataset(AbstractVersionedDataset[str, str]):
    _protocol: str
    _fs: fsspec.AbstractFileSystem
    
    def __init__(self, filepath: str, version: Optional[Version] = None) -> None: ...
    def _describe(self) -> Dict[str, Any]: ...
    def _load(self) -> str: ...
    def _save(self, data: str) -> None: ...
    def _exists(self) -> bool: ...

def my_legacy_dataset(
    filepath_versioned: str,
    save_args: Optional[Any],
    fs_args: Optional[Any],
) -> MyLegacyDataset: ...
def my_legacy_versioned_dataset(
    filepath_versioned: str,
    load_version: Optional[Any],
    save_version: Optional[str],
) -> MyLegacyVersionedDataset: ...

class TestLegacyLoadAndSave:
    def test_saving_none(self, my_legacy_dataset: MyLegacyDataset) -> None: ...
    def test_saving_invalid_data(
        self, my_legacy_dataset: MyLegacyDataset, dummy_data: str
    ) -> None: ...
    @pytest.mark.parametrize("load_version", ["2019-01-01T23.59.59.999Z"], indirect=True)
    @pytest.mark.parametrize("save_version", ["2019-01-02T00.00.00.000Z"], indirect=True)
    def test_save_version_warning(
        self,
        my_legacy_versioned_dataset: MyLegacyVersionedDataset,
        load_version: str,
        save_version: str,
        dummy_data: str,
    ) -> None: ...
    def test_versioning_existing_dataset(
        self,
        my_legacy_dataset: MyLegacyDataset,
        my_legacy_versioned_dataset: MyLegacyVersionedDataset,
        dummy_data: str,
    ) -> None: ...