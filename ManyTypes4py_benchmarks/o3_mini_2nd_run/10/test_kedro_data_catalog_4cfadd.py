from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Pattern, Union

import logging
import re

import pandas as pd
import pytest
from kedro_datasets.pandas import CSVDataset, ParquetDataset
from pandas.testing import assert_frame_equal
from kedro.io import (CachedDataset, DatasetAlreadyExistsError, DatasetError,
                        DatasetNotFoundError, KedroDataCatalog, LambdaDataset, MemoryDataset)
from kedro.io.core import _DEFAULT_PACKAGES, VERSION_FORMAT, Version, VersionAlreadyExistsError, generate_timestamp, parse_dataset_definition


@pytest.fixture
def data_catalog(dataset: Any) -> KedroDataCatalog:
    return KedroDataCatalog(datasets={'test': dataset})


@pytest.fixture
def memory_catalog() -> KedroDataCatalog:
    ds1: MemoryDataset = MemoryDataset({'data': 42})
    ds2: MemoryDataset = MemoryDataset([1, 2, 3, 4, 5])
    return KedroDataCatalog({'ds1': ds1, 'ds2': ds2})


@pytest.fixture
def conflicting_feed_dict() -> Dict[str, Any]:
    return {'ds1': 0, 'ds3': 1}


@pytest.fixture
def multi_catalog() -> KedroDataCatalog:
    csv_1: CSVDataset = CSVDataset(filepath='abc.csv')
    csv_2: CSVDataset = CSVDataset(filepath='def.csv')
    parq: ParquetDataset = ParquetDataset(filepath='xyz.parq')
    return KedroDataCatalog({'abc': csv_1, 'def': csv_2, 'xyz': parq})


@pytest.fixture
def data_catalog_from_config(correct_config: Dict[str, Any]) -> KedroDataCatalog:
    return KedroDataCatalog.from_config(**correct_config)


class TestKedroDataCatalog:
    def test_save_and_load(self, data_catalog: KedroDataCatalog, dummy_dataframe: pd.DataFrame) -> None:
        """Test saving and reloading the dataset"""
        data_catalog.save('test', dummy_dataframe)
        reloaded_df: pd.DataFrame = data_catalog.load('test')
        assert_frame_equal(reloaded_df, dummy_dataframe)

    def test_add_save_and_load(self, dataset: Any, dummy_dataframe: pd.DataFrame) -> None:
        """Test adding and then saving and reloading the dataset"""
        catalog: KedroDataCatalog = KedroDataCatalog(datasets={})
        catalog.add('test', dataset)
        catalog.save('test', dummy_dataframe)
        reloaded_df: pd.DataFrame = catalog.load('test')
        assert_frame_equal(reloaded_df, dummy_dataframe)

    def test_load_error(self, data_catalog: KedroDataCatalog) -> None:
        """Check the error when attempting to load a dataset
        from nonexistent source"""
        pattern: str = 'Failed while loading data from dataset CSVDataset'
        with pytest.raises(DatasetError, match=pattern):
            data_catalog.load('test')

    def test_add_dataset_twice(self, data_catalog: KedroDataCatalog, dataset: Any) -> None:
        """Check the error when attempting to add the dataset twice"""
        pattern: str = "Dataset 'test' has already been registered"
        with pytest.raises(DatasetAlreadyExistsError, match=pattern):
            data_catalog.add('test', dataset)

    def test_load_from_unregistered(self) -> None:
        """Check the error when attempting to load unregistered dataset"""
        catalog: KedroDataCatalog = KedroDataCatalog(datasets={})
        pattern: str = "Dataset 'test' not found in the catalog"
        with pytest.raises(DatasetNotFoundError, match=pattern):
            catalog.load('test')

    def test_save_to_unregistered(self, dummy_dataframe: pd.DataFrame) -> None:
        """Check the error when attempting to save to unregistered dataset"""
        catalog: KedroDataCatalog = KedroDataCatalog(datasets={})
        pattern: str = "Dataset 'test' not found in the catalog"
        with pytest.raises(DatasetNotFoundError, match=pattern):
            catalog.save('test', dummy_dataframe)

    def test_feed_dict(self, memory_catalog: KedroDataCatalog, conflicting_feed_dict: Dict[str, Any]) -> None:
        """Test feed dict overriding some of the datasets"""
        assert 'data' in memory_catalog.load('ds1')
        memory_catalog.add_feed_dict(conflicting_feed_dict, replace=True)
        assert memory_catalog.load('ds1') == 0
        assert isinstance(memory_catalog.load('ds2'), list)
        assert memory_catalog.load('ds3') == 1

    def test_exists(self, data_catalog: KedroDataCatalog, dummy_dataframe: pd.DataFrame) -> None:
        """Test `exists` method invocation"""
        assert not data_catalog.exists('test')
        data_catalog.save('test', dummy_dataframe)
        assert data_catalog.exists('test')

    def test_exists_not_implemented(self, caplog: Any) -> None:
        """Test calling `exists` on the dataset, which didn't implement it"""
        catalog: KedroDataCatalog = KedroDataCatalog(datasets={'test': LambdaDataset(lambda x: x, lambda x: x)})
        result: bool = catalog.exists('test')
        log_record = caplog.records[0]
        assert log_record.levelname == 'WARNING'
        assert "'exists()' not implemented for 'LambdaDataset'. Assuming output does not exist." in log_record.message
        assert result is False

    def test_exists_invalid(self, data_catalog: KedroDataCatalog) -> None:
        """Check the error when calling `exists` on invalid dataset"""
        assert not data_catalog.exists('wrong_key')

    def test_release_unregistered(self, data_catalog: KedroDataCatalog) -> None:
        """Check the error when calling `release` on unregistered dataset"""
        pattern: str = "Dataset \\'wrong_key\\' not found in the catalog"
        with pytest.raises(DatasetNotFoundError, match=pattern) as e:
            data_catalog.release('wrong_key')
        assert 'did you mean' not in str(e.value)

    def test_release_unregistered_typo(self, data_catalog: KedroDataCatalog) -> None:
        """Check the error when calling `release` on mistyped dataset"""
        pattern: str = "Dataset 'text' not found in the catalog - did you mean one of these instead: test"
        with pytest.raises(DatasetNotFoundError, match=re.escape(pattern)):
            data_catalog.release('text')

    def test_multi_catalog_list(self, multi_catalog: KedroDataCatalog) -> None:
        """Test data catalog which contains multiple datasets"""
        entries: List[str] = multi_catalog.list()
        assert 'abc' in entries
        assert 'xyz' in entries

    @pytest.mark.parametrize('pattern,expected', [
        ('^a', ['abc']),
        ('a|x', ['abc', 'xyz']),
        ('^(?!(a|d|x))', []),
        ('def', ['def']),
        ('ghi', []),
        ('', [])
    ])
    def test_multi_catalog_list_regex(self, multi_catalog: KedroDataCatalog, pattern: str, expected: List[str]) -> None:
        """Test that regex patterns filter datasets accordingly"""
        assert multi_catalog.list(regex_search=pattern) == expected

    def test_multi_catalog_list_bad_regex(self, multi_catalog: KedroDataCatalog) -> None:
        """Test that bad regex is caught accordingly"""
        escaped_regex: str = '\\(\\('
        pattern: str = f"Invalid regular expression provided: '{escaped_regex}'"
        with pytest.raises(SyntaxError, match=pattern):
            multi_catalog.list('((')

    @pytest.mark.parametrize('name_regex,type_regex,expected', [
        (re.compile('^a'), None, ['abc']),
        (re.compile('^A'), None, []),
        (re.compile('^A', flags=re.IGNORECASE), None, ['abc']),
        ('a|x', None, ['abc', 'xyz']),
        ('a|d|x', None, ['abc', 'def', 'xyz']),
        ('a|d|x', 'CSVDataset', ['abc', 'def']),
        ('a|d|x', 'kedro_datasets', ['abc', 'def', 'xyz']),
        (None, 'ParquetDataset', ['xyz']),
        ('^(?!(a|d|x))', None, []),
        ('def', None, ['def']),
        (None, None, ['abc', 'def', 'xyz']),
        ('a|d|x', 'no_such_dataset', [])
    ])
    def test_catalog_filter_regex(
        self,
        multi_catalog: KedroDataCatalog,
        name_regex: Optional[Union[str, Pattern]],
        type_regex: Optional[Union[str, Pattern]],
        expected: List[str]
    ) -> None:
        """Test that regex patterns filter materialized datasets accordingly"""
        assert multi_catalog.filter(name_regex=name_regex, type_regex=type_regex) == expected

    @pytest.mark.parametrize('name_regex,type_regex,by_type,expected', [
        ('b|m', None, None, ['boats', 'materialized']),
        (None, None, None, ['boats', 'cars', 'materialized']),
        (None, 'CSVDataset', None, ['boats', 'cars']),
        (None, 'ParquetDataset', None, ['materialized']),
        ('b|c', 'ParquetDataset', None, []),
        (None, None, ParquetDataset, ['materialized']),
        (None, None, [CSVDataset, ParquetDataset], ['boats', 'cars', 'materialized']),
        (None, 'ParquetDataset', [CSVDataset, ParquetDataset], ['materialized']),
        ('b|m', None, [CSVDataset, ParquetDataset], ['boats', 'materialized'])
    ])
    def test_from_config_catalog_filter_regex(
        self,
        data_catalog_from_config: KedroDataCatalog,
        name_regex: Optional[Union[str, Pattern]],
        type_regex: Optional[Union[str, Pattern]],
        by_type: Optional[Union[type, List[type]]],
        expected: List[str]
    ) -> None:
        """Test that regex patterns filter lazy and materialized datasets accordingly"""
        data_catalog_from_config['materialized'] = ParquetDataset(filepath='xyz.parq')
        assert data_catalog_from_config.filter(name_regex=name_regex, type_regex=type_regex, by_type=by_type) == expected

    def test_eq(self, multi_catalog: KedroDataCatalog, data_catalog: KedroDataCatalog) -> None:
        assert multi_catalog == multi_catalog.shallow_copy()
        assert multi_catalog != data_catalog

    def test_datasets_on_init(self, data_catalog_from_config: KedroDataCatalog) -> None:
        """Check datasets are loaded correctly on construction"""
        assert isinstance(data_catalog_from_config.get('boats'), CSVDataset)
        assert isinstance(data_catalog_from_config.get('cars'), CSVDataset)

    def test_datasets_on_add(self, data_catalog_from_config: KedroDataCatalog) -> None:
        """Check datasets are updated correctly after adding"""
        data_catalog_from_config.add('new_dataset', CSVDataset(filepath='some_path'))
        assert isinstance(data_catalog_from_config.get('new_dataset'), CSVDataset)
        assert isinstance(data_catalog_from_config.get('boats'), CSVDataset)

    def test_adding_datasets_not_allowed(self, data_catalog_from_config: KedroDataCatalog) -> None:
        """Check error if user tries to update the datasets attribute"""
        pattern: str = 'Operation not allowed. Please use KedroDataCatalog.add\\(\\) instead.'
        with pytest.raises(AttributeError, match=pattern):
            data_catalog_from_config.datasets = None

    def test_confirm(self, mocker: Any, caplog: Any) -> None:
        """Confirm the dataset"""
        with caplog.at_level(logging.INFO):
            mock_ds: Any = mocker.Mock()
            data_catalog: KedroDataCatalog = KedroDataCatalog(datasets={'mocked': mock_ds})
            data_catalog.confirm('mocked')
            mock_ds.confirm.assert_called_once_with()
            assert caplog.record_tuples == [('kedro.io.kedro_data_catalog', logging.INFO, "Confirming dataset 'mocked'")]

    @pytest.mark.parametrize('dataset_name,error_pattern', [
        ('missing', "Dataset 'missing' not found in the catalog"),
        ('test', "Dataset 'test' does not have 'confirm' method")
    ])
    def test_bad_confirm(self, data_catalog: KedroDataCatalog, dataset_name: str, error_pattern: str) -> None:
        """Test confirming a non-existent dataset or one that
        does not have `confirm` method"""
        with pytest.raises(DatasetError, match=re.escape(error_pattern)):
            data_catalog.confirm(dataset_name)

    def test_shallow_copy_returns_correct_class_type(self) -> None:

        class MyDataCatalog(KedroDataCatalog):
            pass

        data_catalog: MyDataCatalog = MyDataCatalog()
        copy: KedroDataCatalog = data_catalog.shallow_copy()
        assert isinstance(copy, MyDataCatalog)

    @pytest.mark.parametrize(
        'runtime_patterns,sorted_keys_expected',
        [({'{default}': {'type': 'MemoryDataset'}, '{another}#csv': {'type': 'pandas.CSVDataset', 'filepath': 'data/{another}.csv'}}, ['{another}#csv', '{default}'])]
    )
    def test_shallow_copy_adds_patterns(
        self,
        data_catalog: KedroDataCatalog,
        runtime_patterns: Dict[str, Any],
        sorted_keys_expected: List[str]
    ) -> None:
        assert not data_catalog.config_resolver.list_patterns()
        data_catalog = data_catalog.shallow_copy(runtime_patterns)
        assert data_catalog.config_resolver.list_patterns() == sorted_keys_expected

    def test_init_with_raw_data(self, dummy_dataframe: pd.DataFrame, dataset: Any) -> None:
        """Test catalog initialisation with raw data"""
        catalog: KedroDataCatalog = KedroDataCatalog(datasets={'ds': dataset}, raw_data={'df': dummy_dataframe})
        assert 'ds' in catalog
        assert 'df' in catalog
        assert isinstance(catalog['ds'], CSVDataset)
        assert isinstance(catalog['df'], MemoryDataset)

    def test_repr(self, data_catalog_from_config: KedroDataCatalog) -> None:
        assert data_catalog_from_config.__repr__() == str(data_catalog_from_config)

    def test_repr_no_type_found(self, data_catalog_from_config: KedroDataCatalog) -> None:
        del data_catalog_from_config._lazy_datasets['boats'].config['type']
        pattern: str = "'type' is missing from dataset catalog configuration"
        with pytest.raises(DatasetError, match=re.escape(pattern)):
            _ = str(data_catalog_from_config)

    def test_missing_keys_from_load_versions(self, correct_config: Dict[str, Any]) -> None:
        """Test load versions include keys missing in the catalog"""
        pattern: str = "'load_versions' keys [version] are not found in the catalog."
        with pytest.raises(DatasetNotFoundError, match=re.escape(pattern)):
            KedroDataCatalog.from_config(**correct_config, load_versions={'version': 'test_version'})

    def test_get_dataset_matching_pattern(self, data_catalog: KedroDataCatalog) -> None:
        """Test get_dataset() when dataset is not in the catalog but pattern matches"""
        match_pattern_ds: str = 'match_pattern_ds'
        assert match_pattern_ds not in data_catalog
        data_catalog.config_resolver.add_runtime_patterns({'{default}': {'type': 'MemoryDataset'}})
        ds: Any = data_catalog.get_dataset(match_pattern_ds)
        assert isinstance(ds, MemoryDataset)

    def test_remove_runtime_pattern(self, data_catalog: KedroDataCatalog) -> None:
        runtime_pattern: Dict[str, Any] = {'{default}': {'type': 'MemoryDataset'}}
        data_catalog.config_resolver.add_runtime_patterns(runtime_pattern)
        match_pattern_ds: str = 'match_pattern_ds'
        assert match_pattern_ds in data_catalog
        data_catalog.config_resolver.remove_runtime_patterns(runtime_pattern)
        assert match_pattern_ds not in data_catalog

    def test_release(self, data_catalog: KedroDataCatalog) -> None:
        """Test release is called without errors"""
        data_catalog.release('test')

    def test_dataset_property(self, data_catalog_from_config: KedroDataCatalog) -> None:
        """Test _dataset attribute returns the same result as dataset property"""
        _ = data_catalog_from_config['boats']
        assert data_catalog_from_config.datasets == data_catalog_from_config._datasets
        for ds_name in data_catalog_from_config.list():
            assert ds_name in data_catalog_from_config._datasets

    class TestKedroDataCatalogToConfig:
        def test_to_config(
            self,
            correct_config_versioned: Dict[str, Any],
            dataset: Any,
            filepath: str
        ) -> None:
            """Test dumping catalog config"""
            config: Dict[str, Any] = correct_config_versioned['catalog']
            credentials: Dict[str, Any] = correct_config_versioned['credentials']
            catalog: KedroDataCatalog = KedroDataCatalog.from_config(config, credentials)
            catalog['resolved_ds'] = dataset
            catalog['memory_ds'] = [1, 2, 3]
            catalog['params:a.b'] = {'abc': 'def'}
            _ = catalog['cached_ds']
            version: Version = Version(load='fake_load_version.csv', save=None)
            versioned_dataset: CSVDataset = CSVDataset(filepath='shuttles.csv', version=version, metadata=[1, 2, 3])
            cached_versioned_dataset: CachedDataset = CachedDataset(dataset=versioned_dataset)
            catalog['cached_versioned_dataset'] = cached_versioned_dataset
            catalog_config, catalog_credentials, load_version, save_version = catalog.to_config()
            expected_config: Dict[str, Any] = {
                'resolved_ds': {
                    'type': 'kedro_datasets.pandas.csv_dataset.CSVDataset',
                    'filepath': filepath,
                    'save_args': {'index': False},
                    'load_args': None,
                    'credentials': None,
                    'fs_args': None
                },
                'cached_versioned_dataset': {
                    'type': 'kedro.io.cached_dataset.CachedDataset',
                    'copy_mode': None,
                    'versioned': True,
                    'dataset': {
                        'type': 'kedro_datasets.pandas.csv_dataset.CSVDataset',
                        'filepath': 'shuttles.csv',
                        'load_args': None,
                        'save_args': None,
                        'credentials': None,
                        'fs_args': None
                    }
                }
            }
            expected_config.update(config)
            expected_config.pop('parameters', None)
            assert catalog_config == expected_config
            assert catalog_credentials == credentials
            assert catalog._load_versions == {'cached_versioned_dataset': 'fake_load_version.csv'}
            assert catalog._save_version

    class TestKedroDataCatalogFromConfig:
        def test_from_correct_config(
            self,
            data_catalog_from_config: KedroDataCatalog,
            dummy_dataframe: pd.DataFrame
        ) -> None:
            """Test populating the data catalog from config"""
            data_catalog_from_config.save('boats', dummy_dataframe)
            reloaded_df: pd.DataFrame = data_catalog_from_config.load('boats')
            assert_frame_equal(reloaded_df, dummy_dataframe)

        def test_config_missing_type(self, correct_config: Dict[str, Any]) -> None:
            """Check the error if type attribute is missing for some dataset(s)
            in the config"""
            del correct_config['catalog']['boats']['type']
            pattern: str = "An exception occurred when parsing config for dataset 'boats':\n'type' is missing from dataset catalog configuration"
            with pytest.raises(DatasetError, match=re.escape(pattern)):
                KedroDataCatalog.from_config(**correct_config)

        def test_config_invalid_module(self, correct_config: Dict[str, Any]) -> None:
            """Check the error if the type points to nonexistent module"""
            correct_config['catalog']['boats']['type'] = 'kedro.invalid_module_name.io.CSVDataset'
            error_msg: str = "Class 'kedro.invalid_module_name.io.CSVDataset' not found"
            with pytest.raises(DatasetError, match=re.escape(error_msg)):
                KedroDataCatalog.from_config(**correct_config).get('boats')

        def test_config_relative_import(self, correct_config: Dict[str, Any]) -> None:
            """Check the error if the type points to a relative import"""
            correct_config['catalog']['boats']['type'] = '.CSVDatasetInvalid'
            pattern: str = "'type' class path does not support relative paths"
            with pytest.raises(DatasetError, match=re.escape(pattern)):
                KedroDataCatalog.from_config(**correct_config).get('boats')

        def test_config_import_kedro_datasets(self, correct_config: Dict[str, Any], mocker: Any) -> None:
            """Test kedro_datasets default path to the dataset class"""
            import kedro.io.core  # type: ignore
            spy = mocker.spy(kedro.io.core, '_load_obj')
            parse_dataset_definition(correct_config['catalog']['boats'])
            for prefix, call_args in zip(_DEFAULT_PACKAGES, spy.call_args_list):
                assert call_args.args[0] == f'{prefix}pandas.CSVDataset'

        def test_config_missing_class(self, correct_config: Dict[str, Any]) -> None:
            """Check the error if the type points to nonexistent class"""
            correct_config['catalog']['boats']['type'] = 'kedro.io.CSVDatasetInvalid'
            pattern: str = "An exception occurred when parsing config for dataset 'boats':\nClass 'kedro.io.CSVDatasetInvalid' not found, is this a typo?"
            with pytest.raises(DatasetError, match=re.escape(pattern)):
                KedroDataCatalog.from_config(**correct_config).get('boats')

        def test_config_invalid_dataset(self, correct_config: Dict[str, Any]) -> None:
            """Check the error if the type points to invalid class"""
            correct_config['catalog']['boats']['type'] = 'KedroDataCatalog'
            pattern: str = "An exception occurred when parsing config for dataset 'boats':\nDataset type 'kedro.io.kedro_data_catalog.KedroDataCatalog' is invalid: all dataset types must extend 'AbstractDataset'"
            with pytest.raises(DatasetError, match=re.escape(pattern)):
                KedroDataCatalog.from_config(**correct_config).get('boats')

        def test_config_invalid_arguments(self, correct_config: Dict[str, Any]) -> None:
            """Check the error if the dataset config contains invalid arguments"""
            correct_config['catalog']['boats']['save_and_load_args'] = False
            pattern: str = "Dataset 'boats' must only contain arguments valid for the constructor of '.*CSVDataset'"
            with pytest.raises(DatasetError, match=pattern):
                KedroDataCatalog.from_config(**correct_config).get('boats')

        def test_config_invalid_dataset_config(self, correct_config: Dict[str, Any]) -> None:
            correct_config['catalog']['invalid_entry'] = 'some string'
            pattern: str = "Catalog entry 'invalid_entry' is not a valid dataset configuration. \nHint: If this catalog entry is intended for variable interpolation, make sure that the key is preceded by an underscore."
            with pytest.raises(DatasetError, match=pattern):
                KedroDataCatalog.from_config(**correct_config)

        def test_empty_config(self) -> None:
            """Test empty config"""
            assert len(KedroDataCatalog.from_config(None)) == 0

        def test_missing_credentials(self, correct_config: Dict[str, Any]) -> None:
            """Check the error if credentials can't be located"""
            correct_config['catalog']['cars']['credentials'] = 'missing'
            with pytest.raises(KeyError, match="Unable to find credentials \\'missing\\'"):
                KedroDataCatalog.from_config(**correct_config)

        def test_link_credentials(self, correct_config: Dict[str, Any], mocker: Any) -> None:
            """Test credentials being linked to the relevant dataset"""
            mock_client = mocker.patch('kedro_datasets.pandas.csv_dataset.fsspec')
            config: Dict[str, Any] = deepcopy(correct_config)
            del config['catalog']['boats']
            KedroDataCatalog.from_config(**config).get('cars')
            expected_client_kwargs: Dict[str, Any] = correct_config['credentials']['s3_credentials']
            mock_client.filesystem.assert_called_with('s3', **expected_client_kwargs)

        def test_nested_credentials(self, correct_config_with_nested_creds: Dict[str, Any], mocker: Any) -> None:
            mock_client = mocker.patch('kedro_datasets.pandas.csv_dataset.fsspec')
            config: Dict[str, Any] = deepcopy(correct_config_with_nested_creds)
            del config['catalog']['boats']
            KedroDataCatalog.from_config(**config).get('cars')
            expected_client_kwargs: Dict[str, Any] = {
                'client_kwargs': {'credentials': {'client_kwargs': {'aws_access_key_id': 'OTHER_FAKE_ACCESS_KEY', 'aws_secret_access_key': 'OTHER_FAKE_SECRET_KEY'}}},
                'key': 'secret'
            }
            mock_client.filesystem.assert_called_once_with('s3', **expected_client_kwargs)

        def test_missing_nested_credentials(self, correct_config_with_nested_creds: Dict[str, Any]) -> None:
            del correct_config_with_nested_creds['credentials']['other_credentials']
            pattern: str = "Unable to find credentials 'other_credentials'"
            with pytest.raises(KeyError, match=pattern):
                KedroDataCatalog.from_config(**correct_config_with_nested_creds)

        def test_missing_dependency(self, correct_config: Dict[str, Any], mocker: Any) -> None:
            """Test that dependency is missing."""
            pattern: str = 'dependency issue'

            def dummy_load(obj_path: str, *args: Any, **kwargs: Any) -> Any:
                if obj_path == 'kedro_datasets.pandas.CSVDataset':
                    raise AttributeError(pattern)
                if obj_path == 'kedro_datasets.pandas.__all__':
                    return ['CSVDataset']

            mocker.patch('kedro.io.core.load_obj', side_effect=dummy_load)
            with pytest.raises(DatasetError, match=pattern):
                KedroDataCatalog.from_config(**correct_config).get('boats')

        def test_idempotent_catalog(self, correct_config: Dict[str, Any]) -> None:
            """Test that data catalog instantiations are idempotent"""
            _ = KedroDataCatalog.from_config(**correct_config)
            catalog: KedroDataCatalog = KedroDataCatalog.from_config(**correct_config)
            assert catalog

        def test_error_dataset_init(self, bad_config: Dict[str, Any]) -> None:
            """Check the error when trying to instantiate erroneous dataset"""
            pattern: str = "Failed to instantiate dataset \\'bad\\' of type '.*BadDataset'"
            with pytest.raises(DatasetError, match=pattern):
                KedroDataCatalog.from_config(bad_config, None).get('bad')

        def test_validate_dataset_config(self) -> None:
            """Test _validate_dataset_config raises error when wrong dataset config type is passed"""
            pattern: str = "Catalog entry 'bad' is not a valid dataset configuration. \nHint: If this catalog entry is intended for variable interpolation, make sure that the key is preceded by an underscore."
            with pytest.raises(DatasetError, match=pattern):
                KedroDataCatalog._validate_dataset_config(ds_name='bad', ds_config='not_dict')

        def test_confirm(self, tmp_path: Any, caplog: Any, mocker: Any) -> None:
            """Confirm the dataset"""
            with caplog.at_level(logging.INFO):
                mock_confirm = mocker.patch('kedro_datasets.partitions.incremental_dataset.IncrementalDataset.confirm')
                catalog: Dict[str, Any] = {'ds_to_confirm': {'type': 'kedro_datasets.partitions.incremental_dataset.IncrementalDataset', 'dataset': 'pandas.CSVDataset', 'path': str(tmp_path)}}
                data_catalog: KedroDataCatalog = KedroDataCatalog.from_config(catalog=catalog)
                data_catalog.confirm('ds_to_confirm')
                assert caplog.record_tuples == [('kedro.io.kedro_data_catalog', logging.INFO, "Confirming dataset 'ds_to_confirm'")]
                mock_confirm.assert_called_once_with()

        @pytest.mark.parametrize('dataset_name,pattern', [
            ('missing', "Dataset 'missing' not found in the catalog"),
            ('boats', "Dataset 'boats' does not have 'confirm' method")
        ])
        def test_bad_confirm(self, correct_config: Dict[str, Any], dataset_name: str, pattern: str) -> None:
            """Test confirming non existent dataset or the one that
            does not have `confirm` method"""
            data_catalog: KedroDataCatalog = KedroDataCatalog.from_config(**correct_config)
            with pytest.raises(DatasetError, match=re.escape(pattern)):
                data_catalog.confirm(dataset_name)

        def test_iteration(self, correct_config: Dict[str, Any]) -> None:
            """Test iterate through keys, values and items."""
            data_catalog: KedroDataCatalog = KedroDataCatalog.from_config(**correct_config)
            for ds_name_cat, ds_name_config in zip(data_catalog, correct_config['catalog']):
                assert ds_name_cat == ds_name_config
            for ds_name_cat, ds_name_config in zip(data_catalog.keys(), correct_config['catalog']):
                assert ds_name_cat == ds_name_config
            for ds in data_catalog.values():
                assert isinstance(ds, CSVDataset)
            for ds_name, ds in data_catalog.items():
                assert isinstance(ds, CSVDataset)
                assert ds_name in correct_config['catalog']

        def test_getitem_setitem(self, correct_config: Dict[str, Any]) -> None:
            """Test get and set item."""
            data_catalog: KedroDataCatalog = KedroDataCatalog.from_config(**correct_config)
            data_catalog['test'] = 123
            assert isinstance(data_catalog['test'], MemoryDataset)

        def test_ipython_key_completions(self, correct_config: Dict[str, Any]) -> None:
            data_catalog: KedroDataCatalog = KedroDataCatalog.from_config(**correct_config)
            assert data_catalog._ipython_key_completions_() == list(correct_config['catalog'].keys())

    class TestDataCatalogVersioned:
        def test_from_correct_config_versioned(
            self,
            correct_config: Dict[str, Any],
            dummy_dataframe: pd.DataFrame
        ) -> None:
            """Test load and save of versioned datasets from config"""
            correct_config['catalog']['boats']['versioned'] = True
            current_ts: datetime = datetime.now(tz=timezone.utc)
            fmt: str = '{d.year:04d}-{d.month:02d}-{d.day:02d}T{d.hour:02d}.{d.minute:02d}.{d.second:02d}.{ms:03d}Z'
            version: str = fmt.format(d=current_ts, ms=current_ts.microsecond // 1000)
            catalog: KedroDataCatalog = KedroDataCatalog.from_config(**correct_config, load_versions={'boats': version}, save_version=version)
            catalog.save('boats', dummy_dataframe)
            path: Path = Path(correct_config['catalog']['boats']['filepath'])
            path = path / version / path.name
            assert path.is_file()
            reloaded_df: pd.DataFrame = catalog.load('boats')
            assert_frame_equal(reloaded_df, dummy_dataframe)
            reloaded_df_version: pd.DataFrame = catalog.load('boats', version=version)
            assert_frame_equal(reloaded_df_version, dummy_dataframe)
            actual_timestamp: datetime = datetime.strptime(catalog.datasets['boats'].resolve_load_version(), VERSION_FORMAT)
            expected_timestamp: datetime = current_ts.replace(microsecond=current_ts.microsecond // 1000 * 1000, tzinfo=None)
            assert actual_timestamp == expected_timestamp

        @pytest.mark.parametrize('versioned', [True, False])
        def test_from_correct_config_versioned_warn(self, caplog: Any, correct_config: Dict[str, Any], versioned: bool) -> None:
            """Check the warning if `version` attribute was added
            to the dataset config"""
            correct_config['catalog']['boats']['versioned'] = versioned
            correct_config['catalog']['boats']['version'] = True
            KedroDataCatalog.from_config(**correct_config).get('boats')
            log_record = caplog.records[0]
            expected_log_message: str = "'version' attribute removed from dataset configuration since it is a reserved word and cannot be directly specified"
            assert log_record.levelname == 'WARNING'
            assert expected_log_message in log_record.message

        def test_from_correct_config_load_versions_warn(self, correct_config: Dict[str, Any]) -> None:
            correct_config['catalog']['boats']['versioned'] = True
            version: str = generate_timestamp()
            load_version: Dict[str, str] = {'non-boat': version}
            pattern: str = "\\'load_versions\\' keys \\[non-boat\\] are not found in the catalog\\."
            with pytest.raises(DatasetNotFoundError, match=pattern):
                KedroDataCatalog.from_config(**correct_config, load_versions=load_version)

        def test_compare_tracking_and_other_dataset_versioned(
            self,
            correct_config_with_tracking_ds: Dict[str, Any],
            dummy_dataframe: pd.DataFrame
        ) -> None:
            """Test saving of tracking datasets from config results in the same
            save version as other versioned datasets."""
            catalog: KedroDataCatalog = KedroDataCatalog.from_config(**correct_config_with_tracking_ds)
            catalog.save('boats', dummy_dataframe)
            dummy_data: Dict[str, Any] = {'col1': 1, 'col2': 2, 'col3': 3}
            catalog.save('planes', dummy_data)
            csv_timestamp: datetime = datetime.strptime(catalog.datasets['boats'].resolve_save_version(), VERSION_FORMAT)
            tracking_timestamp: datetime = datetime.strptime(catalog.datasets['planes'].resolve_save_version(), VERSION_FORMAT)
            assert tracking_timestamp == csv_timestamp

        def test_load_version(self, correct_config: Dict[str, Any], dummy_dataframe: pd.DataFrame, mocker: Any) -> None:
            """Test load versioned datasets from config"""
            new_dataframe: pd.DataFrame = pd.DataFrame({'col1': [0, 0], 'col2': [0, 0], 'col3': [0, 0]})
            correct_config['catalog']['boats']['versioned'] = True
            mocker.patch('kedro.io.kedro_data_catalog.generate_timestamp', side_effect=['first', 'second'])
            catalog: KedroDataCatalog = KedroDataCatalog.from_config(**correct_config)
            catalog.save('boats', dummy_dataframe)
            catalog = KedroDataCatalog.from_config(**correct_config)
            catalog.save('boats', new_dataframe)
            assert_frame_equal(catalog.load('boats', version='first'), dummy_dataframe)
            assert_frame_equal(catalog.load('boats', version='second'), new_dataframe)
            assert_frame_equal(catalog.load('boats'), new_dataframe)

        def test_load_version_on_unversioned_dataset(self, correct_config: Dict[str, Any], dummy_dataframe: pd.DataFrame, mocker: Any) -> None:
            mocker.patch('kedro.io.kedro_data_catalog.generate_timestamp', return_value='first')
            catalog: KedroDataCatalog = KedroDataCatalog.from_config(**correct_config)
            catalog.save('boats', dummy_dataframe)
            with pytest.raises(DatasetError):
                catalog.load('boats', version='first')

        def test_redefine_save_version_via_catalog(self, correct_config: Dict[str, Any], dataset_versioned: Any) -> None:
            """Test redefining save version when it is already set"""
            catalog: KedroDataCatalog = KedroDataCatalog.from_config(**correct_config)
            with pytest.raises(VersionAlreadyExistsError):
                catalog['ds_versioned'] = dataset_versioned
            correct_config['catalog']['boats']['versioned'] = True
            catalog = KedroDataCatalog.from_config(**correct_config)
            with pytest.raises(VersionAlreadyExistsError):
                catalog['ds_versioned'] = dataset_versioned

        def test_set_load_and_save_versions(self, correct_config: Dict[str, Any], dataset_versioned: Any) -> None:
            """Test setting load and save versions for catalog based on dataset's versions provided"""
            catalog: KedroDataCatalog = KedroDataCatalog(datasets={'ds_versioned': dataset_versioned})
            assert catalog._load_versions['ds_versioned'] == dataset_versioned._version.load
            assert catalog._save_version == dataset_versioned._version.save

        def test_set_same_versions(self, correct_config: Dict[str, Any], dataset_versioned: Any) -> None:
            """Test setting the same load and save versions for catalog based on dataset's versions provided"""
            catalog: KedroDataCatalog = KedroDataCatalog(datasets={'ds_versioned': dataset_versioned})
            catalog['ds_same_versions'] = dataset_versioned
            assert catalog._load_versions['ds_versioned'] == dataset_versioned._version.load
            assert catalog._save_version == dataset_versioned._version.save

        def test_redefine_load_version(self, correct_config: Dict[str, Any], dataset_versioned: Any) -> None:
            """Test redefining save version when it is already set"""
            catalog: KedroDataCatalog = KedroDataCatalog(datasets={'ds_versioned': dataset_versioned})
            dataset_versioned._version = Version(load='another_load_version.csv', save='test_save_version.csv')
            catalog['ds_same_versions'] = dataset_versioned
            assert catalog._load_versions['ds_same_versions'] == dataset_versioned._version.load
            assert catalog._load_versions['ds_versioned'] == 'test_load_version.csv'
            assert catalog._save_version == dataset_versioned._version.save

        def test_redefine_save_version(self, correct_config: Dict[str, Any], dataset_versioned: Any) -> None:
            """Test redefining save version when it is already set"""
            catalog: KedroDataCatalog = KedroDataCatalog(datasets={'ds_versioned': dataset_versioned})
            dataset_versioned._version = Version(load='another_load_version.csv', save='another_save_version.csv')
            with pytest.raises(VersionAlreadyExistsError):
                catalog['ds_same_versions'] = dataset_versioned

        def test_redefine_save_version_with_cached_dataset(self, correct_config: Dict[str, Any], cached_dataset_versioned: Any) -> None:
            """Test redefining load and save version with CachedDataset"""
            catalog: KedroDataCatalog = KedroDataCatalog.from_config(**correct_config)
            with pytest.raises(VersionAlreadyExistsError):
                catalog['cached_dataset_versioned'] = cached_dataset_versioned
            cached_dataset_versioned._dataset._version = Version(load='test_load_version.csv', save=None)
            catalog['cached_dataset_versioned'] = cached_dataset_versioned
            assert catalog._load_versions['cached_dataset_versioned'] == 'test_load_version.csv'
            assert catalog._save_version

# End of annotated code.
