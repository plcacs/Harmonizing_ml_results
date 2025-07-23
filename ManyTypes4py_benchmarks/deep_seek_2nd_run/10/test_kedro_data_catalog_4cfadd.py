import logging
import re
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Pattern, Tuple, Type, Union, cast

import pandas as pd
import pytest
from kedro_datasets.pandas import CSVDataset, ParquetDataset
from pandas.testing import assert_frame_equal
from kedro.io import (
    CachedDataset,
    DatasetAlreadyExistsError,
    DatasetError,
    DatasetNotFoundError,
    KedroDataCatalog,
    LambdaDataset,
    MemoryDataset,
)
from kedro.io.core import (
    _DEFAULT_PACKAGES,
    VERSION_FORMAT,
    Version,
    VersionAlreadyExistsError,
    generate_timestamp,
    parse_dataset_definition,
)

@pytest.fixture
def data_catalog(dataset: CSVDataset) -> KedroDataCatalog:
    return KedroDataCatalog(datasets={"test": dataset})


@pytest.fixture
def memory_catalog() -> KedroDataCatalog:
    ds1 = MemoryDataset({"data": 42})
    ds2 = MemoryDataset([1, 2, 3, 4, 5])
    return KedroDataCatalog({"ds1": ds1, "ds2": ds2})


@pytest.fixture
def conflicting_feed_dict() -> Dict[str, int]:
    return {"ds1": 0, "ds3": 1}


@pytest.fixture
def multi_catalog() -> KedroDataCatalog:
    csv_1 = CSVDataset(filepath="abc.csv")
    csv_2 = CSVDataset(filepath="def.csv")
    parq = ParquetDataset(filepath="xyz.parq")
    return KedroDataCatalog({"abc": csv_1, "def": csv_2, "xyz": parq})


@pytest.fixture
def data_catalog_from_config(correct_config: Dict[str, Any]) -> KedroDataCatalog:
    return KedroDataCatalog.from_config(**correct_config)


class TestKedroDataCatalog:
    def test_save_and_load(
        self, data_catalog: KedroDataCatalog, dummy_dataframe: pd.DataFrame
    ) -> None:
        """Test saving and reloading the dataset"""
        data_catalog.save("test", dummy_dataframe)
        reloaded_df = data_catalog.load("test")
        assert_frame_equal(reloaded_df, dummy_dataframe)

    def test_add_save_and_load(
        self, dataset: CSVDataset, dummy_dataframe: pd.DataFrame
    ) -> None:
        """Test adding and then saving and reloading the dataset"""
        catalog = KedroDataCatalog(datasets={})
        catalog.add("test", dataset)
        catalog.save("test", dummy_dataframe)
        reloaded_df = catalog.load("test")
        assert_frame_equal(reloaded_df, dummy_dataframe)

    def test_load_error(self, data_catalog: KedroDataCatalog) -> None:
        """Check the error when attempting to load a dataset
        from nonexistent source"""
        pattern = "Failed while loading data from dataset CSVDataset"
        with pytest.raises(DatasetError, match=pattern):
            data_catalog.load("test")

    def test_add_dataset_twice(
        self, data_catalog: KedroDataCatalog, dataset: CSVDataset
    ) -> None:
        """Check the error when attempting to add the dataset twice"""
        pattern = "Dataset 'test' has already been registered"
        with pytest.raises(DatasetAlreadyExistsError, match=pattern):
            data_catalog.add("test", dataset)

    def test_load_from_unregistered(self) -> None:
        """Check the error when attempting to load unregistered dataset"""
        catalog = KedroDataCatalog(datasets={})
        pattern = "Dataset 'test' not found in the catalog"
        with pytest.raises(DatasetNotFoundError, match=pattern):
            catalog.load("test")

    def test_save_to_unregistered(self, dummy_dataframe: pd.DataFrame) -> None:
        """Check the error when attempting to save to unregistered dataset"""
        catalog = KedroDataCatalog(datasets={})
        pattern = "Dataset 'test' not found in the catalog"
        with pytest.raises(DatasetNotFoundError, match=pattern):
            catalog.save("test", dummy_dataframe)

    def test_feed_dict(
        self, memory_catalog: KedroDataCatalog, conflicting_feed_dict: Dict[str, int]
    ) -> None:
        """Test feed dict overriding some of the datasets"""
        assert "data" in memory_catalog.load("ds1")
        memory_catalog.add_feed_dict(conflicting_feed_dict, replace=True)
        assert memory_catalog.load("ds1") == 0
        assert isinstance(memory_catalog.load("ds2"), list)
        assert memory_catalog.load("ds3") == 1

    def test_exists(
        self, data_catalog: KedroDataCatalog, dummy_dataframe: pd.DataFrame
    ) -> None:
        """Test `exists` method invocation"""
        assert not data_catalog.exists("test")
        data_catalog.save("test", dummy_dataframe)
        assert data_catalog.exists("test")

    def test_exists_not_implemented(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test calling `exists` on the dataset, which didn't implement it"""
        catalog = KedroDataCatalog(datasets={"test": LambdaDataset(None, None)})
        result = catalog.exists("test")
        log_record = caplog.records[0]
        assert log_record.levelname == "WARNING"
        assert (
            "'exists()' not implemented for 'LambdaDataset'. Assuming output does not exist."
            in log_record.message
        )
        assert result is False

    def test_exists_invalid(self, data_catalog: KedroDataCatalog) -> None:
        """Check the error when calling `exists` on invalid dataset"""
        assert not data_catalog.exists("wrong_key")

    def test_release_unregistered(self, data_catalog: KedroDataCatalog) -> None:
        """Check the error when calling `release` on unregistered dataset"""
        pattern = "Dataset \\'wrong_key\\' not found in the catalog"
        with pytest.raises(DatasetNotFoundError, match=pattern) as e:
            data_catalog.release("wrong_key")
        assert "did you mean" not in str(e.value)

    def test_release_unregistered_typo(self, data_catalog: KedroDataCatalog) -> None:
        """Check the error when calling `release` on mistyped dataset"""
        pattern = (
            "Dataset 'text' not found in the catalog - did you mean one of these instead: test"
        )
        with pytest.raises(DatasetNotFoundError, match=re.escape(pattern)):
            data_catalog.release("text")

    def test_multi_catalog_list(self, multi_catalog: KedroDataCatalog) -> None:
        """Test data catalog which contains multiple datasets"""
        entries = multi_catalog.list()
        assert "abc" in entries
        assert "xyz" in entries

    @pytest.mark.parametrize(
        "pattern,expected",
        [
            ("^a", ["abc"]),
            ("a|x", ["abc", "xyz"]),
            ("^(?!(a|d|x))", []),
            ("def", ["def"]),
            ("ghi", []),
            ("", []),
        ],
    )
    def test_multi_catalog_list_regex(
        self, multi_catalog: KedroDataCatalog, pattern: str, expected: List[str]
    ) -> None:
        """Test that regex patterns filter datasets accordingly"""
        assert multi_catalog.list(regex_search=pattern) == expected

    def test_multi_catalog_list_bad_regex(self, multi_catalog: KedroDataCatalog) -> None:
        """Test that bad regex is caught accordingly"""
        escaped_regex = "\\(\\("
        pattern = f"Invalid regular expression provided: '{escaped_regex}'"
        with pytest.raises(SyntaxError, match=pattern):
            multi_catalog.list("((")

    @pytest.mark.parametrize(
        "name_regex,type_regex,expected",
        [
            (re.compile("^a"), None, ["abc"]),
            (re.compile("^A"), None, []),
            (re.compile("^A", flags=re.IGNORECASE), None, ["abc"]),
            ("a|x", None, ["abc", "xyz"]),
            ("a|d|x", None, ["abc", "def", "xyz"]),
            ("a|d|x", "CSVDataset", ["abc", "def"]),
            ("a|d|x", "kedro_datasets", ["abc", "def", "xyz"]),
            (None, "ParquetDataset", ["xyz"]),
            ("^(?!(a|d|x))", None, []),
            ("def", None, ["def"]),
            (None, None, ["abc", "def", "xyz"]),
            ("a|d|x", "no_such_dataset", []),
        ],
    )
    def test_catalog_filter_regex(
        self,
        multi_catalog: KedroDataCatalog,
        name_regex: Union[str, Pattern[str], None],
        type_regex: Optional[str],
        expected: List[str],
    ) -> None:
        """Test that regex patterns filter materialized datasets accordingly"""
        assert (
            multi_catalog.filter(name_regex=name_regex, type_regex=type_regex)
            == expected
        )

    @pytest.mark.parametrize(
        "name_regex,type_regex,by_type,expected",
        [
            ("b|m", None, None, ["boats", "materialized"]),
            (None, None, None, ["boats", "cars", "materialized"]),
            (None, "CSVDataset", None, ["boats", "cars"]),
            (None, "ParquetDataset", None, ["materialized"]),
            ("b|c", "ParquetDataset", None, []),
            (None, None, ParquetDataset, ["materialized"]),
            (
                None,
                None,
                [CSVDataset, ParquetDataset],
                ["boats", "cars", "materialized"],
            ),
            (
                None,
                "ParquetDataset",
                [CSVDataset, ParquetDataset],
                ["materialized"],
            ),
            ("b|m", None, [CSVDataset, ParquetDataset], ["boats", "materialized"]),
        ],
    )
    def test_from_config_catalog_filter_regex(
        self,
        data_catalog_from_config: KedroDataCatalog,
        name_regex: Optional[str],
        type_regex: Optional[str],
        by_type: Optional[Union[Type[Any], List[Type[Any]]],
        expected: List[str],
    ) -> None:
        """Test that regex patterns filter lazy and materialized datasets accordingly"""
        data_catalog_from_config["materialized"] = ParquetDataset(filepath="xyz.parq")
        assert (
            data_catalog_from_config.filter(
                name_regex=name_regex, type_regex=type_regex, by_type=by_type
            )
            == expected
        )

    def test_eq(
        self, multi_catalog: KedroDataCatalog, data_catalog: KedroDataCatalog
    ) -> None:
        assert multi_catalog == multi_catalog.shallow_copy()
        assert multi_catalog != data_catalog

    def test_datasets_on_init(self, data_catalog_from_config: KedroDataCatalog) -> None:
        """Check datasets are loaded correctly on construction"""
        assert isinstance(data_catalog_from_config.get("boats"), CSVDataset)
        assert isinstance(data_catalog_from_config.get("cars"), CSVDataset)

    def test_datasets_on_add(self, data_catalog_from_config: KedroDataCatalog) -> None:
        """Check datasets are updated correctly after adding"""
        data_catalog_from_config.add(
            "new_dataset", CSVDataset(filepath="some_path")
        )
        assert isinstance(data_catalog_from_config.get("new_dataset"), CSVDataset)
        assert isinstance(data_catalog_from_config.get("boats"), CSVDataset)

    def test_adding_datasets_not_allowed(
        self, data_catalog_from_config: KedroDataCatalog
    ) -> None:
        """Check error if user tries to update the datasets attribute"""
        pattern = (
            "Operation not allowed. Please use KedroDataCatalog.add\\(\\) instead."
        )
        with pytest.raises(AttributeError, match=pattern):
            data_catalog_from_config.datasets = None

    def test_confirm(self, mocker: Any, caplog: pytest.LogCaptureFixture) -> None:
        """Confirm the dataset"""
        with caplog.at_level(logging.INFO):
            mock_ds = mocker.Mock()
            data_catalog = KedroDataCatalog(datasets={"mocked": mock_ds})
            data_catalog.confirm("mocked")
            mock_ds.confirm.assert_called_once_with()
            assert caplog.record_tuples == [
                (
                    "kedro.io.kedro_data_catalog",
                    logging.INFO,
                    "Confirming dataset 'mocked'",
                )
            ]

    @pytest.mark.parametrize(
        "dataset_name,error_pattern",
        [
            ("missing", "Dataset 'missing' not found in the catalog"),
            ("test", "Dataset 'test' does not have 'confirm' method"),
        ],
    )
    def test_bad_confirm(
        self,
        data_catalog: KedroDataCatalog,
        dataset_name: str,
        error_pattern: str,
    ) -> None:
        """Test confirming a non-existent dataset or one that
        does not have `confirm` method"""
        with pytest.raises(DatasetError, match=re.escape(error_pattern)):
            data_catalog.confirm(dataset_name)

    def test_shallow_copy_returns_correct_class_type(self) -> None:
        class MyDataCatalog(KedroDataCatalog):
            pass

        data_catalog = MyDataCatalog()
        copy = data_catalog.shallow_copy()
        assert isinstance(copy, MyDataCatalog)

    @pytest.mark.parametrize(
        "runtime_patterns,sorted_keys_expected",
        [
            (
                {
                    "{default}": {"type": "MemoryDataset"},
                    "{another}#csv": {
                        "type": "pandas.CSVDataset",
                        "filepath": "data/{another}.csv",
                    },
                },
                ["{another}#csv", "{default}"],
            )
        ],
    )
    def test_shallow_copy_adds_patterns(
        self,
        data_catalog: KedroDataCatalog,
        runtime_patterns: Dict[str, Dict[str, str]],
        sorted_keys_expected: List[str],
    ) -> None:
        assert not data_catalog.config_resolver.list_patterns()
        data_catalog = data_catalog.shallow_copy(runtime_patterns)
        assert data_catalog.config_resolver.list_patterns() == sorted_keys_expected

    def test_init_with_raw_data(
        self, dummy_dataframe: pd.DataFrame, dataset: CSVDataset
    ) -> None:
        """Test catalog initialisation with raw data"""
        catalog = KedroDataCatalog(
            datasets={"ds": dataset}, raw_data={"df": dummy_dataframe}
        )
        assert "ds" in catalog
        assert "df" in catalog
        assert isinstance(catalog["ds"], CSVDataset)
        assert isinstance(catalog["df"], MemoryDataset)

    def test_repr(self, data_catalog_from_config: KedroDataCatalog) -> None:
        assert data_catalog_from_config.__repr__() == str(data_catalog_from_config)

    def test_repr_no_type_found(
        self, data_catalog_from_config: KedroDataCatalog
    ) -> None:
        del data_catalog_from_config._lazy_datasets["boats"].config["type"]
        pattern = "'type' is missing from dataset catalog configuration"
        with pytest.raises(DatasetError, match=re.escape(pattern)):
            _ = str(data_catalog_from_config)

    def test_missing_keys_from_load_versions(
        self, correct_config: Dict[str, Any]
    ) -> None:
        """Test load versions include keys missing in the catalog"""
        pattern = "'load_versions' keys [version] are not found in the catalog."
        with pytest.raises(DatasetNotFoundError, match=re.escape(pattern)):
            KedroDataCatalog.from_config(
                **correct_config, load_versions={"version": "test_version"}
            )

    def test_get_dataset_matching_pattern(
        self, data_catalog: KedroDataCatalog
    ) -> None:
        """Test get_dataset() when dataset is not in the catalog but pattern matches"""
        match_pattern_ds = "match_pattern_ds"
        assert match_pattern_ds not in data_catalog
        data_catalog.config_resolver.add_runtime_patterns(
            {"{default}": {"type": "MemoryDataset"}}
        )
        ds = data_catalog.get_dataset(match_pattern_ds)
        assert isinstance(ds, MemoryDataset)

    def test_remove_runtime_pattern(self, data_catalog: KedroDataCatalog) -> None:
        runtime_pattern = {"{default}": {"type": "MemoryDataset"}}
        data_catalog.config_resolver.add_runtime_patterns(runtime_pattern)
        match_pattern_ds = "match_pattern_ds"
        assert match_pattern_ds in data_catalog
        data_catalog.config_resolver.remove_runtime_patterns(runtime_pattern)
        assert match_pattern_ds not in data_catalog

    def test_release(self, data_catalog: KedroDataCatalog) -> None:
        """Test release is called without errors"""
        data_catalog.release("test")

    def test_dataset_property(
        self, data_catalog_from_config: KedroDataCatalog
    ) -> None:
        """Test _dataset attribute returns the same result as dataset property"""
        _ = data_catalog_from_config["boats"]
        assert data_catalog_from_config.datasets == data_catalog_from_config._datasets
        for ds_name in data_catalog_from_config.list():
            assert ds_name in data_catalog_from_config._datasets

    class TestKedroDataCatalogToConfig:
        def test_to_config(
            self,
            correct_config_versioned: Dict[str, Any],
            dataset: CSVDataset,
            filepath: str,
        ) -> None:
            """Test dumping catalog config"""
            config = correct_config_versioned["catalog"]
            credentials = correct_config_versioned["credentials"]
            catalog = KedroDataCatalog.from_config(config, credentials)
            catalog["resolved_ds"] = dataset
            catalog["memory_ds"] = [1, 2, 3]
            catalog["params:a.b"] = {"abc