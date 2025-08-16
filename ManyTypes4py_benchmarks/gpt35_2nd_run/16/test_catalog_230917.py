from typing import Dict, Any, List
import pytest
import yaml
from click.testing import CliRunner
from kedro_datasets.pandas import CSVDataset
from kedro.io import DataCatalog, KedroDataCatalog, MemoryDataset
from kedro.pipeline import node
from kedro.pipeline.modular_pipeline import pipeline as modular_pipeline

def identity(data: Any) -> Any:
    return data

@pytest.fixture
def fake_load_context(mocker: Any) -> Any:
    context = mocker.MagicMock()
    return mocker.patch('kedro.framework.session.KedroSession.load_context', return_value=context)

PIPELINE_NAME: str = 'pipeline'

@pytest.fixture
def mock_pipelines(mocker: Any) -> Any:
    dummy_pipelines: Dict[str, Any] = {PIPELINE_NAME: modular_pipeline([]), 'second': modular_pipeline([])}
    return mocker.patch('kedro.framework.cli.catalog.pipelines', dummy_pipelines)

@pytest.fixture()
def fake_credentials_config(tmp_path: Any) -> Dict[str, Any]:
    return {'db_connection': {'con': 'foo'}}

@pytest.fixture
def fake_catalog_config() -> Dict[str, Any]:
    config: Dict[str, Any] = {'parquet_{factory_pattern}': {'type': 'pandas.ParquetDataset', 'filepath': 'data/01_raw/{factory_pattern}.parquet', 'credentials': 'db_connection'}, 'csv_{factory_pattern}': {'type': 'pandas.CSVDataset', 'filepath': 'data/01_raw/{factory_pattern}.csv'}, 'csv_test': {'type': 'pandas.CSVDataset', 'filepath': 'test.csv'}}
    return config

@pytest.fixture
def fake_catalog_config_resolved() -> Dict[str, Any]:
    config: Dict[str, Any] = {'parquet_example': {'type': 'pandas.ParquetDataset', 'filepath': 'data/01_raw/example.parquet', 'credentials': {'con': 'foo'}}, 'csv_example': {'type': 'pandas.CSVDataset', 'filepath': 'data/01_raw/example.csv'}, 'csv_test': {'type': 'pandas.CSVDataset', 'filepath': 'test.csv'}}
    return config

@pytest.fixture
def fake_catalog_with_overlapping_factories() -> Dict[str, Any]:
    config: Dict[str, Any] = {'an_example_dataset': {'type': 'pandas.CSVDataset', 'filepath': 'dummy_filepath'}, 'an_example_{placeholder}': {'type': 'dummy_type', 'filepath': 'dummy_filepath'}, 'an_example_{place}_{holder}': {'type': 'dummy_type', 'filepath': 'dummy_filepath'}, 'on_{example_placeholder}': {'type': 'dummy_type', 'filepath': 'dummy_filepath'}, 'an_{example_placeholder}': {'type': 'dummy_type', 'filepath': 'dummy_filepath'}}
    return config

@pytest.fixture
def fake_catalog_config_with_factories(fake_metadata: Any) -> Dict[str, Any]:
    config: Dict[str, Any] = {'parquet_{factory_pattern}': {'type': 'pandas.ParquetDataset', 'filepath': 'data/01_raw/{factory_pattern}.parquet'}, 'csv_{factory_pattern}': {'type': 'pandas.CSVDataset', 'filepath': 'data/01_raw/{factory_pattern}.csv'}, 'explicit_ds': {'type': 'pandas.CSVDataset', 'filepath': 'test.csv'}, '{factory_pattern}_ds': {'type': 'pandas.ParquetDataset', 'filepath': 'data/01_raw/{factory_pattern}_ds.parquet'}, 'partitioned_{factory_pattern}': {'type': 'partitions.PartitionedDataset', 'path': 'data/01_raw', 'dataset': 'pandas.CSVDataset', 'metadata': {'my-plugin': {'path': 'data/01_raw'}}}}
    return config

@pytest.fixture
def fake_catalog_config_with_factories_resolved() -> Dict[str, Any]:
    config: Dict[str, Any] = {'parquet_example': {'type': 'pandas.ParquetDataset', 'filepath': 'data/01_raw/example.parquet'}, 'csv_example': {'type': 'pandas.CSVDataset', 'filepath': 'data/01_raw/example.csv'}, 'explicit_ds': {'type': 'pandas.CSVDataset', 'filepath': 'test.csv'}, 'partitioned_example': {'type': 'partitions.PartitionedDataset', 'path': 'data/01_raw', 'dataset': 'pandas.CSVDataset', 'metadata': {'my-plugin': {'path': 'data/01_raw'}}}}
    return config

@pytest.mark.usefixtures('chdir_to_dummy_project', 'fake_load_context', 'mock_pipelines')
class TestCatalogListCommand:

    def test_list_all_pipelines(self, fake_project_cli_parametrized: Any, fake_metadata: Any, mocker: Any) -> None:
        ...

    def test_list_specific_pipelines(self, fake_project_cli_parametrized: Any, fake_metadata: Any, mocker: Any) -> None:
        ...

    def test_not_found_pipeline(self, fake_project_cli_parametrized: Any, fake_metadata: Any) -> None:
        ...

    @pytest.mark.parametrize('catalog_type', [DataCatalog, KedroDataCatalog])
    def test_no_param_datasets_in_respose(self, fake_project_cli_parametrized: Any, fake_metadata: Any, fake_load_context: Any, mocker: Any, mock_pipelines: Any, catalog_type: Any) -> None:
        ...

    @pytest.mark.parametrize('catalog_type', [DataCatalog, KedroDataCatalog])
    def test_default_dataset(self, fake_project_cli_parametrized: Any, fake_metadata: Any, fake_load_context: Any, mocker: Any, mock_pipelines: Any, catalog_type: Any) -> None:
        ...

    @pytest.mark.parametrize('catalog_type', [DataCatalog, KedroDataCatalog])
    def test_list_factory_generated_datasets(self, fake_project_cli_parametrized: Any, fake_metadata: Any, fake_load_context: Any, mocker: Any, mock_pipelines: Any, fake_catalog_config: Any, fake_credentials_config: Any, catalog_type: Any) -> None:
        ...

@pytest.mark.usefixtures('chdir_to_dummy_project', 'fake_load_context')
class TestCatalogCreateCommand:

    def test_pipeline_argument_is_required(self, fake_project_cli_parametrized: Any) -> None:
        ...

    def test_not_found_pipeline(self, fake_project_cli_parametrized: Any, fake_metadata: Any, mock_pipelines: Any) -> None:
        ...

    def test_catalog_is_created_in_base_by_default(self, fake_project_cli_parametrized: Any, fake_metadata: Any, fake_repo_path: Any, catalog_path: Any) -> None:
        ...

    def test_catalog_is_created_in_correct_env(self, fake_project_cli_parametrized: Any, fake_metadata: Any, catalog_path: Any) -> None:
        ...

    @pytest.mark.parametrize('catalog_type', [DataCatalog, KedroDataCatalog])
    def test_no_missing_datasets(self, fake_project_cli_parametrized: Any, fake_metadata: Any, fake_load_context: Any, fake_repo_path: Any, mock_pipelines: Any, catalog_type: Any) -> None:
        ...

    @pytest.mark.usefixtures('fake_repo_path')
    def test_missing_datasets_appended(self, fake_project_cli: Any, fake_metadata: Any, catalog_path: Any) -> None:
        ...

    def test_bad_env(self, fake_project_cli: Any, fake_metadata: Any) -> None:
        ...

@pytest.mark.usefixtures('chdir_to_dummy_project', 'fake_load_context')
class TestCatalogFactoryCommands:

    @pytest.mark.parametrize('catalog_type', [DataCatalog, KedroDataCatalog])
    @pytest.mark.usefixtures('mock_pipelines')
    def test_rank_catalog_factories(self, fake_project_cli: Any, fake_metadata: Any, mocker: Any, fake_load_context: Any, fake_catalog_with_overlapping_factories: Any, catalog_type: Any) -> None:
        ...

    @pytest.mark.parametrize('catalog_type', [DataCatalog, KedroDataCatalog])
    def test_rank_catalog_factories_with_no_factories(self, fake_project_cli: Any, fake_metadata: Any, fake_load_context: Any, catalog_type: Any) -> None:
        ...

    @pytest.mark.parametrize('catalog_type', [DataCatalog, KedroDataCatalog])
    @pytest.mark.usefixtures('mock_pipelines')
    def test_catalog_resolve(self, fake_project_cli: Any, fake_metadata: Any, fake_load_context: Any, mocker: Any, mock_pipelines: Any, fake_catalog_config: Any, fake_catalog_config_resolved: Any, fake_credentials_config: Any, catalog_type: Any) -> None:
        ...

    @pytest.mark.parametrize('catalog_type', [DataCatalog, KedroDataCatalog])
    @pytest.mark.usefixtures('mock_pipelines')
    def test_catalog_resolve_nested_config(self, fake_project_cli: Any, fake_metadata: Any, fake_load_context: Any, mocker: Any, mock_pipelines: Any, fake_catalog_config_with_factories: Any, fake_catalog_config_with_factories_resolved: Any, catalog_type: Any) -> None:
        ...

    @pytest.mark.parametrize('catalog_type', [DataCatalog, KedroDataCatalog])
    @pytest.mark.usefixtures('mock_pipelines')
    def test_no_param_datasets_in_resolve(self, fake_project_cli: Any, fake_metadata: Any, fake_load_context: Any, mocker: Any, mock_pipelines: Any, catalog_type: Any) -> None:
        ...
