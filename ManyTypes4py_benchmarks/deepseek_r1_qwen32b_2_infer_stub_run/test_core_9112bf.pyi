from __future__ import annotations
from decimal import Decimal
from fractions import Fraction
from pathlib import PurePosixPath
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Union,
    Tuple,
    Iterator,
    Callable,
    Type,
    Pattern,
    cast,
)
import logging
import pytest
import shutil
from datetime import datetime
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
from typing import Any
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
from pytest import fixture

FALSE_BUILTINS = List[Union[bool, int, float, complex, Decimal, Fraction, str, tuple, list, dict, set, range]]

class MyDataset(AbstractDataset):
    def __init__(self, filepath: str = '', save_args: Optional[dict] = None, fs_args: Optional[dict] = None, var: Any = None) -> None:
        ...

    def _describe(self) -> Dict[str, Any]:
        ...

    def _exists(self) -> bool:
        ...

    def _load(self) -> pd.DataFrame:
        ...

    def _save(self, data: Any) -> None:
        ...

class MyVersionedDataset(AbstractVersionedDataset[str, str]):
    def __init__(self, filepath: str, version: Optional[Version] = None) -> None:
        ...

    def _describe(self) -> Dict[str, Any]:
        ...

    def _load(self) -> str:
        ...

    def _save(self, data: str) -> None:
        ...

    def _exists(self) -> bool:
        ...

class MyLocalVersionedDataset(AbstractVersionedDataset[str, str]):
    def __init__(self, filepath: str, version: Optional[Version] = None) -> None:
        ...

    def _describe(self) -> Dict[str, Any]:
        ...

    def _load(self) -> str:
        ...

    def _save(self, data: str) -> None:
        ...

    def _exists(self) -> bool:
        ...

class MyOtherVersionedDataset(MyLocalVersionedDataset):
    def _exists(self) -> bool:
        ...

@fixture(params=[None])
def load_version(request) -> Optional[Any]:
    ...

@fixture(params=[None])
def save_version(request) -> Optional[Any]:
    ...

@fixture(params=[None])
def load_args(request) -> Optional[Any]:
    ...

@fixture(params=[None])
def save_args(request) -> Optional[Any]:
    ...

@fixture(params=[None])
def fs_args(request) -> Optional[Any]:
    ...

@fixture
def filepath_versioned(tmp_path: Path) -> str:
    ...

@fixture
def my_dataset(filepath_versioned: str, save_args: Optional[Any], fs_args: Optional[Any]) -> MyDataset:
    ...

@fixture
def my_versioned_dataset(filepath_versioned: str, load_version: Optional[Any], save_version: Optional[Any]) -> MyVersionedDataset:
    ...

@fixture
def dummy_data() -> str:
    ...

class TestCoreFunctions:
    @pytest.mark.parametrize('var', [1, True, *FALSE_BUILTINS])
    def test_str_representation(self, var: Union[int, bool, Decimal, Fraction, str]) -> None:
        ...

    def test_str_representation_none(self) -> None:
        ...

    @pytest.mark.parametrize('describe_return', [None, {'key_1': 'val_1', 2: 'val_2'}])
    def test_repr_bad_describe(self, describe_return: Optional[Dict[Any, Any]], caplog: pytest.LogCaptureFixture) -> None:
        ...

    def test_get_filepath_str(self) -> None:
        ...

    @pytest.mark.parametrize('filepath,expected_result', [('s3://bucket/file.txt', ('s3', 'bucket/file.txt')), ...])
    def test_get_protocol_and_path(self, filepath: str, expected_result: Tuple[str, str]) -> None:
        ...

    @pytest.mark.parametrize('filepath', ['http://example.com/file.txt', 'https://example.com/file.txt'])
    def test_get_protocol_and_path_http_with_version(self, filepath: str) -> None:
        ...

    @pytest.mark.parametrize('input', [{'key1': 'invalid value'}, {'key2': 'invalid;value'}])
    def test_validate_forbidden_chars(self, input: Dict[str, str]) -> None:
        ...

    def test_dataset_name_typo(self, mocker: pytest_mock.MockFixture) -> None:
        ...

    def test_dataset_missing_dependencies(self, mocker: pytest_mock.MockFixture) -> None:
        ...

    def test_parse_dataset_definition(self) -> None:
        ...

    def test_parse_dataset_definition_with_python_class_type(self) -> None:
        ...

    def test_load_and_save_are_wrapped_once(self) -> None:
        ...

class TestAbstractVersionedDataset:
    def test_version_str_repr(self, load_version: Optional[Any], save_version: Optional[Any]) -> None:
        ...

    def test_save_and_load(self, my_versioned_dataset: MyVersionedDataset, dummy_data: str) -> None:
        ...

    def test_resolve_save_version(self, dummy_data: str) -> None:
        ...

    def test_no_versions(self, my_versioned_dataset: MyVersionedDataset) -> None:
        ...

    def test_local_exists(self, dummy_data: str) -> None:
        ...

    def test_exists_general_exception(self) -> None:
        ...

    def test_exists(self, my_versioned_dataset: MyVersionedDataset, dummy_data: str) -> None:
        ...

    def test_prevent_overwrite(self, my_versioned_dataset: MyVersionedDataset, dummy_data: str) -> None:
        ...

    @pytest.mark.parametrize('load_version', ['2019-01-01T23.59.59.999Z'], indirect=True)
    @pytest.mark.parametrize('save_version', ['2019-01-02T00.00.00.000Z'], indirect=True)
    def test_save_version_warning(self, my_versioned_dataset: MyVersionedDataset, load_version: str, save_version: str, dummy_data: str) -> None:
        ...

    def test_versioning_existing_dataset(self, my_dataset: MyDataset, my_versioned_dataset: MyVersionedDataset, dummy_data: str) -> None:
        ...

    def test_cache_release(self, my_versioned_dataset: MyVersionedDataset) -> None:
        ...

class MyLegacyDataset(AbstractDataset):
    def __init__(self, filepath: str = '', save_args: Optional[dict] = None, fs_args: Optional[dict] = None, var: Any = None) -> None:
        ...

    def _describe(self) -> Dict[str, Any]:
        ...

    def _exists(self) -> bool:
        ...

    def _load(self) -> pd.DataFrame:
        ...

    def _save(self, data: Any) -> None:
        ...

class MyLegacyVersionedDataset(AbstractVersionedDataset[str, str]):
    def __init__(self, filepath: str, version: Optional[Version] = None) -> None:
        ...

    def _describe(self) -> Dict[str, Any]:
        ...

    def _load(self) -> str:
        ...

    def _save(self, data: str) -> None:
        ...

    def _exists(self) -> bool:
        ...

@fixture
def my_legacy_dataset(filepath_versioned: str, save_args: Optional[Any], fs_args: Optional[Any]) -> MyLegacyDataset:
    ...

@fixture
def my_legacy_versioned_dataset(filepath_versioned: str, load_version: Optional[Any], save_version: Optional[Any]) -> MyLegacyVersionedDataset:
    ...

class TestLegacyLoadAndSave:
    def test_saving_none(self, my_legacy_dataset: MyLegacyDataset) -> None:
        ...

    def test_saving_invalid_data(self, my_legacy_dataset: MyLegacyDataset, dummy_data: str) -> None:
        ...

    @pytest.mark.parametrize('load_version', ['2019-01-01T23.59.59.999Z'], indirect=True)
    @pytest.mark.parametrize('save_version', ['2019-01-02T00.00.00.000Z'], indirect=True)
    def test_save_version_warning(self, my_legacy_versioned_dataset: MyLegacyVersionedDataset, load_version: str, save_version: str, dummy_data: str) -> None:
        ...

    def test_versioning_existing_dataset(self, my_legacy_dataset: MyLegacyDataset, my_legacy_versioned_dataset: MyLegacyVersionedDataset, dummy_data: str) -> None:
        ...