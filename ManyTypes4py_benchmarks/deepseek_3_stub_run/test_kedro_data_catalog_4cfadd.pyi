import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Pattern, Union, overload
from typing_extensions import Self
import pandas as pd
import pytest
from kedro_datasets.pandas import CSVDataset, ParquetDataset
from kedro.io import (
    CachedDataset,
    DatasetAlreadyExistsError,
    DatasetError,
    DatasetNotFoundError,
    KedroDataCatalog,
    LambdaDataset,
    MemoryDataset,
)
from kedro.io.core import VERSION_FORMAT, Version, VersionAlreadyExistsError

@pytest.fixture
def data_catalog(dataset: CSVDataset) -> KedroDataCatalog:
    ...

@pytest.fixture
def memory_catalog() -> KedroDataCatalog:
    ...

@pytest.fixture
def conflicting_feed_dict() -> Dict[str, int]:
    ...

@pytest.fixture
def multi_catalog() -> KedroDataCatalog:
    ...

@pytest.fixture
def data_catalog_from_config(correct_config: Dict[str, Any]) -> KedroDataCatalog:
    ...

class TestKedroDataCatalog:
    def test_save_and_load(
        self, data_catalog: KedroDataCatalog, dummy_dataframe: pd.DataFrame
    ) -> None:
        ...

    def test_add_save_and_load(
        self, dataset: CSVDataset, dummy_dataframe: pd.DataFrame
    ) -> None:
        ...

    def test_load_error(self, data_catalog: KedroDataCatalog) -> None:
        ...

    def test_add_dataset_twice(
        self, data_catalog: KedroDataCatalog, dataset: CSVDataset
    ) -> None:
        ...

    def test_load_from_unregistered(self) -> None:
        ...

    def test_save_to_unregistered(self, dummy_dataframe: pd.DataFrame) -> None:
        ...

    def test_feed_dict(
        self,
        memory_catalog: KedroDataCatalog,
        conflicting_feed_dict: Dict[str, int],
    ) -> None:
        ...

    def test_exists(
        self, data_catalog: KedroDataCatalog, dummy_dataframe: pd.DataFrame
    ) -> None:
        ...

    def test_exists_not_implemented(self, caplog: pytest.LogCaptureFixture) -> None:
        ...

    def test_exists_invalid(self, data_catalog: KedroDataCatalog) -> None:
        ...

    def test_release_unregistered(self, data_catalog: KedroDataCatalog) -> None:
        ...

    def test_release_unregistered_typo(self, data_catalog: KedroDataCatalog) -> None:
        ...

    def test_multi_catalog_list(self, multi_catalog: KedroDataCatalog) -> None:
        ...

    @pytest.mark.parametrize("pattern,expected", ...)
    def test_multi_catalog_list_regex(
        self,
        multi_catalog: KedroDataCatalog,
        pattern: str,
        expected: List[str],
    ) -> None:
        ...

    def test_multi_catalog_list_bad_regex(
        self, multi_catalog: KedroDataCatalog
    ) -> None:
        ...

    @pytest.mark.parametrize("name_regex,type_regex,expected", ...)
    def test_catalog_filter_regex(
        self,
        multi_catalog: KedroDataCatalog,
        name_regex: Optional[Union[str, Pattern[str]]],
        type_regex: Optional[Union[str, Pattern[str]]],
        expected: List[str],
    ) -> None:
        ...

    @pytest.mark.parametrize("name_regex,type_regex,by_type,expected", ...)
    def test_from_config_catalog_filter_regex(
        self,
        data_catalog_from_config: KedroDataCatalog,
        name_regex: Optional[Union[str, Pattern[str]]],
        type_regex: Optional[Union[str, Pattern[str]]],
        by_type: Optional[Union[type, List[type]]],
        expected: List[str],
    ) -> None:
        ...

    def test_eq(
        self, multi_catalog: KedroDataCatalog, data_catalog: KedroDataCatalog
    ) -> None:
        ...

    def test_datasets_on_init(
        self, data_catalog_from_config: KedroDataCatalog
    ) -> None:
        ...

    def test_datasets_on_add(self, data_catalog_from_config: KedroDataCatalog) -> None:
        ...

    def test_adding_datasets_not_allowed(
        self, data_catalog_from_config: KedroDataCatalog
    ) -> None:
        ...

    def test_confirm(
        self, mocker: pytest.MockerFixture, caplog: pytest.LogCaptureFixture
    ) -> None:
        ...

    @pytest.mark.parametrize("dataset_name,error_pattern", ...)
    def test_bad_confirm(
        self,
        data_catalog: KedroDataCatalog,
        dataset_name: str,
        error_pattern: str,
    ) -> None:
        ...

    def test_shallow_copy_returns_correct_class_type(self) -> None:
        ...

    @pytest.mark.parametrize("runtime_patterns,sorted_keys_expected", ...)
    def test_shallow_copy_adds_patterns(
        self,
        data_catalog: KedroDataCatalog,
        runtime_patterns: Dict[str, Dict[str, Any]],
        sorted_keys_expected: List[str],
    ) -> None:
        ...

    def test_init_with_raw_data(
        self, dummy_dataframe: pd.DataFrame, dataset: CSVDataset
    ) -> None:
        ...

    def test_repr(self, data_catalog_from_config: KedroDataCatalog) -> None:
        ...

    def test_repr_no_type_found(
        self, data_catalog_from_config: KedroDataCatalog
    ) -> None:
        ...

    def test_missing_keys_from_load_versions(
        self, correct_config: Dict[str, Any]
    ) -> None:
        ...

    def test_get_dataset_matching_pattern(
        self, data_catalog: KedroDataCatalog
    ) -> None:
        ...

    def test_remove_runtime_pattern(self, data_catalog: KedroDataCatalog) -> None:
        ...

    def test_release(self, data_catalog: KedroDataCatalog) -> None:
        ...

    def test_dataset_property(self, data_catalog_from_config: KedroDataCatalog) -> None:
        ...

    class TestKedroDataCatalogToConfig:
        def test_to_config(
            self,
            correct_config_versioned: Dict[str, Any],
            dataset: CSVDataset,
            filepath: str,
        ) -> None:
            ...

    class TestKedroDataCatalogFromConfig:
        def test_from_correct_config(
            self,
            data_catalog_from_config: KedroDataCatalog,
            dummy_dataframe: pd.DataFrame,
        ) -> None:
            ...

        def test_config_missing_type(
            self, correct_config: Dict[str, Any]
        ) -> None:
            ...

        def test_config_invalid_module(
            self, correct_config: Dict[str, Any]
        ) -> None:
            ...

        def test_config_relative_import(
            self, correct_config: Dict[str, Any]
        ) -> None:
            ...

        def test_config_import_kedro_datasets(
            self, correct_config: Dict[str, Any], mocker: pytest.MockerFixture
        ) -> None:
            ...

        def test_config_missing_class(
            self, correct_config: Dict[str, Any]
        ) -> None:
            ...

        def test_config_invalid_dataset(
            self, correct_config: Dict[str, Any]
        ) -> None:
            ...

        def test_config_invalid_arguments(
            self, correct_config: Dict[str, Any]
        ) -> None:
            ...

        def test_config_invalid_dataset_config(
            self, correct_config: Dict[str, Any]
        ) -> None:
            ...

        def test_empty_config(self) -> None:
            ...

        def test_missing_credentials(
            self, correct_config: Dict[str, Any]
        ) -> None:
            ...

        def test_link_credentials(
            self, correct_config: Dict[str, Any], mocker: pytest.MockerFixture
        ) -> None:
            ...

        def test_nested_credentials(
            self,
            correct_config_with_nested_creds: Dict[str, Any],
            mocker: pytest.MockerFixture,
        ) -> None:
            ...

        def test_missing_nested_credentials(
            self, correct_config_with_nested_creds: Dict[str, Any]
        ) -> None:
            ...

        def test_missing_dependency(
            self, correct_config: Dict[str, Any], mocker: pytest.MockerFixture
        ) -> None:
            ...

        def test_idempotent_catalog(
            self, correct_config: Dict[str, Any]
        ) -> None:
            ...

        def test_error_dataset_init(
            self, bad_config: Dict[str, Any]
        ) -> None:
            ...

        def test_validate_dataset_config(self) -> None:
            ...

        def test_confirm(
            self,
            tmp_path: Path,
            caplog: pytest.LogCaptureFixture,
            mocker: pytest.MockerFixture,
        ) -> None:
            ...

        @pytest.mark.parametrize("dataset_name,pattern", ...)
        def test_bad_confirm(
            self,
            correct_config: Dict[str, Any],
            dataset_name: str,
            pattern: str,
        ) -> None:
            ...

        def test_iteration(self, correct_config: Dict[str, Any]) -> None:
            ...

        def test_getitem_setitem(self, correct_config: Dict[str, Any]) -> None:
            ...

        def test_ipython_key_completions(
            self, correct_config: Dict[str, Any]
        ) -> None:
            ...

    class TestDataCatalogVersioned:
        def test_from_correct_config_versioned(
            self,
            correct_config: Dict[str, Any],
            dummy_dataframe: pd.DataFrame,
        ) -> None:
            ...

        @pytest.mark.parametrize("versioned", ...)
        def test_from_correct_config_versioned_warn(
            self,
            caplog: pytest.LogCaptureFixture,
            correct_config: Dict[str, Any],
            versioned: bool,
        ) -> None:
            ...

        def test_from_correct_config_load_versions_warn(
            self, correct_config: Dict[str, Any]
        ) -> None:
            ...

        def test_compare_tracking_and_other_dataset_versioned(
            self,
            correct_config_with_tracking_ds: Dict[str, Any],
            dummy_dataframe: pd.DataFrame,
        ) -> None:
            ...

        def test_load_version(
            self,
            correct_config: Dict[str, Any],
            dummy_dataframe: pd.DataFrame,
            mocker: pytest.MockerFixture,
        ) -> None:
            ...

        def test_load_version_on_unversioned_dataset(
            self,
            correct_config: Dict[str, Any],
            dummy_dataframe: pd.DataFrame,
            mocker: pytest.MockerFixture,
        ) -> None:
            ...

        def test_redefine_save_version_via_catalog(
            self,
            correct_config: Dict[str, Any],
            dataset_versioned: CSVDataset,
        ) -> None:
            ...

        def test_set_load_and_save_versions(
            self,
            correct_config: Dict[str, Any],
            dataset_versioned: CSVDataset,
        ) -> None:
            ...

        def test_set_same_versions(
            self,
            correct_config: Dict[str, Any],
            dataset_versioned: CSVDataset,
        ) -> None:
            ...

        def test_redefine_load_version(
            self,
            correct_config: Dict[str, Any],
            dataset_versioned: CSVDataset,
        ) -> None:
            ...

        def test_redefine_save_version(
            self,
            correct_config: Dict[str, Any],
            dataset_versioned: CSVDataset,
        ) -> None:
            ...

        def test_redefine_save_version_with_cached_dataset(
            self,
            correct_config: Dict[str, Any],
            cached_dataset_versioned: CachedDataset,
        ) -> None:
            ...