import pytest
import yaml
from click.testing import CliRunner
from kedro_datasets.pandas import CSVDataset
from kedro.io import DataCatalog, KedroDataCatalog, MemoryDataset
from kedro.pipeline import node
from kedro.pipeline.modular_pipeline import pipeline as modular_pipeline
from typing import Dict, Any, List, Set, Union, Optional, MagicMock, Callable
from pathlib import Path

@pytest.fixture
def fake_load_context(mocker: Any) -> MagicMock:
    context = mocker.MagicMock()
    return mocker.patch('kedro.framework.session.KedroSession.load_context', return_value=context)

PIPELINE_NAME: str = 'pipeline'

@pytest.fixture
def mock_pipelines(mocker: Any) -> Dict[str, Any]:
    dummy_pipelines: Dict[str, Any] = {PIPELINE_NAME: modular_pipeline([]), 'second': modular_pipeline([])}
    return mocker.patch('kedro.framework.cli.catalog.pipelines', dummy_pipelines)

@pytest.fixture()
def fake_credentials_config(tmp_path: Path) -> Dict[str, Dict[str, str]]:
    return {'db_connection': {'con': 'foo'}}

@pytest.fixture
def fake_catalog_config() -> Dict[str, Dict[str, Any]]:
    config: Dict[str, Dict[str, Any]] = {'parquet_{factory_pattern}': {'type': 'pandas.ParquetDataset', 'filepath': 'data/01_raw/{factory_pattern}.parquet', 'credentials': 'db_connection'}, 'csv_{factory_pattern}': {'type': 'pandas.CSVDataset', 'filepath': 'data/01_raw/{factory_pattern}.csv'}, 'csv_test': {'type': 'pandas.CSVDataset', 'filepath': 'test.csv'}}
    return config

@pytest.fixture
def fake_catalog_config_resolved() -> Dict[str, Dict[str, Any]]:
    config: Dict[str, Dict[str, Any]] = {'parquet_example': {'type': 'pandas.ParquetDataset', 'filepath': 'data/01_raw/example.parquet', 'credentials': {'con': 'foo'}}, 'csv_example': {'type': 'pandas.CSVDataset', 'filepath': 'data/01_raw/example.csv'}, 'csv_test': {'type': 'pandas.CSVDataset', 'filepath': 'test.csv'}}
    return config

@pytest.fixture
def fake_catalog_with_overlapping_factories() -> Dict[str, Dict[str, Any]]:
    config: Dict[str, Dict[str, Any]] = {'an_example_dataset': {'type': 'pandas.CSVDataset', 'filepath': 'dummy_filepath'}, 'an_example_{placeholder}': {'type': 'dummy_type', 'filepath': 'dummy_filepath'}, 'an_example_{place}_{holder}': {'type': 'dummy_type', 'filepath': 'dummy_filepath'}, 'on_{example_placeholder}': {'type': 'dummy_type', 'filepath': 'dummy_filepath'}, 'an_{example_placeholder}': {'type': 'dummy_type', 'filepath': 'dummy_filepath'}}
    return config

@pytest.fixture
def fake_catalog_config_with_factories(fake_metadata: Any) -> Dict[str, Dict[str, Any]]:
    config: Dict[str, Dict[str, Any]] = {'parquet_{factory_pattern}': {'type': 'pandas.ParquetDataset', 'filepath': 'data/01_raw/{factory_pattern}.parquet'}, 'csv_{factory_pattern}': {'type': 'pandas.CSVDataset', 'filepath': 'data/01_raw/{factory_pattern}.csv'}, 'explicit_ds': {'type': 'pandas.CSVDataset', 'filepath': 'test.csv'}, '{factory_pattern}_ds': {'type': 'pandas.ParquetDataset', 'filepath': 'data/01_raw/{factory_pattern}_ds.parquet'}, 'partitioned_{factory_pattern}': {'type': 'partitions.PartitionedDataset', 'path': 'data/01_raw', 'dataset': 'pandas.CSVDataset', 'metadata': {'my-plugin': {'path': 'data/01_raw'}}}}
    return config

@pytest.fixture
def fake_catalog_config_with_factories_resolved() -> Dict[str, Dict[str, Any]]:
    config: Dict[str, Dict[str, Any]] = {'parquet_example': {'type': 'pandas.ParquetDataset', 'filepath': 'data/01_raw/example.parquet'}, 'csv_example': {'type': 'pandas.CSVDataset', 'filepath': 'data/01_raw/example.csv'}, 'explicit_ds': {'type': 'pandas.CSVDataset', 'filepath': 'test.csv'}, 'partitioned_example': {'type': 'partitions.PartitionedDataset', 'path': 'data/01_raw', 'dataset': 'pandas.CSVDataset', 'metadata': {'my-plugin': {'path': 'data/01_raw'}}}}
    return config

@pytest.mark.usefixtures('chdir_to_dummy_project', 'fake_load_context', 'mock_pipelines')
class TestCatalogListCommand:

    def test_list_all_pipelines(self, fake_project_cli_parametrized: Any, fake_metadata: Any, mocker: Any) -> None:
        yaml_dump_mock = mocker.patch('yaml.dump', return_value='Result YAML')
        result = CliRunner().invoke(fake_project_cli_parametrized, ['catalog', 'list'], obj=fake_metadata)
        assert not result.exit_code
        expected_dict: Dict[str, Dict[str, Any]] = {"Datasets in 'pipeline' pipeline": {}, "Datasets in 'second' pipeline": {}}
        yaml_dump_mock.assert_called_once_with(expected_dict)

    def test_list_specific_pipelines(self, fake_project_cli_parametrized: Any, fake_metadata: Any, mocker: Any) -> None:
        yaml_dump_mock = mocker.patch('yaml.dump', return_value='Result YAML')
        result = CliRunner().invoke(fake_project_cli_parametrized, ['catalog', 'list', '--pipeline', PIPELINE_NAME], obj=fake_metadata)
        assert not result.exit_code
        expected_dict: Dict[str, Dict[str, Any]] = {f"Datasets in '{PIPELINE_NAME}' pipeline": {}}
        yaml_dump_mock.assert_called_once_with(expected_dict)

    def test_not_found_pipeline(self, fake_project_cli_parametrized: Any, fake_metadata: Any) -> None:
        result = CliRunner().invoke(fake_project_cli_parametrized, ['catalog', 'list', '--pipeline', 'fake'], obj=fake_metadata)
        assert result.exit_code
        expected_output = "Error: 'fake' pipeline not found! Existing pipelines: pipeline, second"
        assert expected_output in result.output

    @pytest.mark.parametrize('catalog_type', [DataCatalog, KedroDataCatalog])
    def test_no_param_datasets_in_respose(self, fake_project_cli_parametrized: Any, fake_metadata: Any, fake_load_context: MagicMock, mocker: Any, mock_pipelines: Dict[str, Any], catalog_type: Any) -> None:
        yaml_dump_mock = mocker.patch('yaml.dump', return_value='Result YAML')
        mocked_context = fake_load_context.return_value
        catalog_datasets: Dict[str, Any] = {'iris_data': CSVDataset(filepath='test.csv'), 'intermediate': MemoryDataset(), 'parameters': MemoryDataset(), 'params:data_ratio': MemoryDataset(), 'not_used': CSVDataset(filepath='test2.csv')}
        mocked_context.catalog = catalog_type(datasets=catalog_datasets)
        mocker.patch.object(mock_pipelines[PIPELINE_NAME], 'datasets', return_value=catalog_datasets.keys() - {'not_used'})
        result = CliRunner().invoke(fake_project_cli_parametrized, ['catalog', 'list'], obj=fake_metadata)
        assert not result.exit_code
        expected_dict: Dict[str, Dict[str, Dict[str, List[str]]]] = {f"Datasets in '{PIPELINE_NAME}' pipeline": {'Datasets mentioned in pipeline': {'CSVDataset': ['iris_data'], 'MemoryDataset': ['intermediate']}, 'Datasets not mentioned in pipeline': {'CSVDataset': ['not_used']}}}
        key = f"Datasets in '{PIPELINE_NAME}' pipeline"
        assert yaml_dump_mock.call_count == 1
        assert yaml_dump_mock.call_args[0][0][key] == expected_dict[key]

    @pytest.mark.parametrize('catalog_type', [DataCatalog, KedroDataCatalog])
    def test_default_dataset(self, fake_project_cli_parametrized: Any, fake_metadata: Any, fake_load_context: MagicMock, mocker: Any, mock_pipelines: Dict[str, Any], catalog_type: Any) -> None:
        """Test that datasets that are found in `Pipeline.datasets()`,
        but not in the catalog, are outputted under the key "DefaultDataset".
        """
        yaml_dump_mock = mocker.patch('yaml.dump', return_value='Result YAML')
        mocked_context = fake_load_context.return_value
        catalog_datasets: Dict[str, Any] = {'some_dataset': CSVDataset(filepath='test.csv')}
        mocked_context.catalog = catalog_type(datasets=catalog_datasets)
        mocker.patch.object(mock_pipelines[PIPELINE_NAME], 'datasets', return_value=catalog_datasets.keys() | {'intermediate'})
        result = CliRunner().invoke(fake_project_cli_parametrized, ['catalog', 'list'], obj=fake_metadata)
        assert not result.exit_code
        expected_dict: Dict[str, Dict[str, Dict[str, List[str]]]] = {f"Datasets in '{PIPELINE_NAME}' pipeline": {'Datasets mentioned in pipeline': {'CSVDataset': ['some_dataset'], 'DefaultDataset': ['intermediate']}}}
        key = f"Datasets in '{PIPELINE_NAME}' pipeline"
        assert yaml_dump_mock.call_count == 1
        assert yaml_dump_mock.call_args[0][0][key] == expected_dict[key]

    @pytest.mark.parametrize('catalog_type', [DataCatalog, KedroDataCatalog])
    def test_list_factory_generated_datasets(self, fake_project_cli_parametrized: Any, fake_metadata: Any, fake_load_context: MagicMock, mocker: Any, mock_pipelines: Dict[str, Any], fake_catalog_config: Dict[str, Dict[str, Any]], fake_credentials_config: Dict[str, Dict[str, str]], catalog_type: Any) -> None:
        """Test that datasets generated from factory patterns in the catalog
        are resolved correctly under the correct dataset classes.
        """
        yaml_dump_mock = mocker.patch('yaml.dump', return_value='Result YAML')
        mocked_context = fake_load_context.return_value
        mocked_context.catalog = catalog_type.from_config(catalog=fake_catalog_config, credentials=fake_credentials_config)
        mocker.patch.object(mock_pipelines[PIPELINE_NAME], 'datasets', return_value=mocked_context.catalog._datasets.keys() | {'csv_example', 'parquet_example'})
        result = CliRunner().invoke(fake_project_cli_parametrized, ['catalog', 'list'], obj=fake_metadata)
        assert not result.exit_code
        expected_dict: Dict[str, Dict[str, Dict[str, List[str]]]] = {f"Datasets in '{PIPELINE_NAME}' pipeline": {'Datasets generated from factories': {'pandas.CSVDataset': ['csv_example'], 'pandas.ParquetDataset': ['parquet_example']}, 'Datasets mentioned in pipeline': {'CSVDataset': ['csv_test']}}}
        key = f"Datasets in '{PIPELINE_NAME}' pipeline"
        assert yaml_dump_mock.call_count == 1
        assert yaml_dump_mock.call_args[0][0][key] == expected_dict[key]

def identity(data: Any) -> Any:
    return data

@pytest.mark.usefixtures('chdir_to_dummy_project')
class TestCatalogCreateCommand:
    PIPELINE_NAME: str = 'data_engineering'

    @staticmethod
    @pytest.fixture(params=['base'])
    def catalog_path(request: Any, fake_repo_path: Path) -> Path:
        catalog_path = fake_repo_path / 'conf' / request.param
        yield catalog_path
        for file in catalog_path.glob('catalog_*'):
            file.unlink()

    def test_pipeline_argument_is_required(self, fake_project_cli_parametrized: Any) -> None:
        result = CliRunner().invoke(fake_project_cli_parametrized, ['catalog', 'create'])
        assert result.exit_code
        expected_output = "Error: Missing option '--pipeline' / '-p'."
        assert expected_output in result.output

    @pytest.mark.usefixtures('fake_load_context')
    def test_not_found_pipeline(self, fake_project_cli_parametrized: Any, fake_metadata: Any, mock_pipelines: Dict[str, Any]) -> None:
        result = CliRunner().invoke(fake_project_cli_parametrized, ['catalog', 'create', '--pipeline', 'fake'], obj=fake_metadata)
        assert result.exit_code
        existing_pipelines = ', '.join(sorted(mock_pipelines.keys()))
        expected_output = f"Error: 'fake' pipeline not found! Existing pipelines: {existing_pipelines}\n"
        assert expected_output in result.output

    def test_catalog_is_created_in_base_by_default(self, fake_project_cli_parametrized: Any, fake_metadata: Any, fake_repo_path: Path, catalog_path: Path) -> None:
        main_catalog_path = fake_repo_path / 'conf' / 'base' / 'catalog.yml'
        main_catalog_config = yaml.safe_load(main_catalog_path.read_text())
        assert 'example_iris_data' in main_catalog_config
        data_catalog_file = catalog_path / f'catalog_{self.PIPELINE_NAME}.yml'
        result = CliRunner().invoke(fake_project_cli_parametrized, ['catalog', 'create', '--pipeline', self.PIPELINE_NAME], obj=fake_metadata)
        assert not result.exit_code
        assert data_catalog_file.is_file()
        expected_catalog_config: Dict[str, Dict[str, str]] = {'example_test_x': {'type': 'MemoryDataset'}, 'example_test_y': {'type': 'MemoryDataset'}, 'example_train_x': {'type': 'MemoryDataset'}, 'example_train_y': {'type': 'MemoryDataset'}}
        catalog_config = yaml.safe_load(data_catalog_file.read_text())
        assert catalog_config == expected_catalog_config

    @pytest.mark.parametrize('catalog_path', ['local'], indirect=True)
    def test_catalog_is_created_in_correct_env(self, fake_project_cli_parametrized: Any, fake_metadata: Any, catalog_path: Path) -> None:
        data_catalog_file = catalog_path / f'catalog_{self.PIPELINE_NAME}.yml'
        env = catalog_path.name
        result = CliRunner().invoke(fake_project_cli_parametrized, ['catalog', 'create', '--pipeline', self.PIPELINE_NAME, '--env', env], obj=fake_metadata)
        assert not result.exit_code
        assert data_catalog_file.is_file()

    @pytest.mark.parametrize('catalog_type', [DataCatalog, KedroDataCatalog])
    def test_no_missing_datasets(self, fake_project_cli_parametrized: Any, fake_metadata: Any, fake_load_context: MagicMock, fake_repo_path: Path, mock_pipelines: Dict[str, Any], catalog_type: Any) -> None:
        mocked_context = fake_load_context.return_value
        catalog_datasets: Dict[str, Any] = {'input_data': CSVDataset(filepath='test.csv'), 'output_data': CSVDataset(filepath='test2.csv')}
        mocked_context.catalog = catalog_type(datasets=catalog_datasets)
        mocked_context.project_path = fake_repo_path
        mock_pipelines[self.PIPELINE_NAME] = modular_pipeline([node(identity, 'input_data', 'output_data')])
        data_catalog_file = fake_repo_path / 'conf' / 'base' / f'catalog_{self.PIPELINE_NAME}.yml'
        result = CliRunner().invoke(fake_project_cli_parametrized, ['catalog', 'create', '--pipeline', self.PIPELINE_NAME], obj=fake_metadata)
        assert not result.exit_code
        assert not data_catalog_file.exists()

    @pytest.mark.usefixtures('fake_repo_path')
    def test_missing_datasets_appended(self, fake_project_cli: Any, fake_metadata: Any, catalog_path: Path) -> None:
        data_catalog_file = catalog_path / f'catalog_{self.PIPELINE_NAME}.yml'
        catalog_config: Dict[str, Dict[str, str]] = {'example_test_x': {'type': 'pandas.CSVDataset', 'filepath': 'test.csv'}}
        with data_catalog_file.open(mode='w') as catalog_file:
            yaml.safe_dump(catalog_config, catalog_file, default_flow_style=False)
        result = CliRunner().invoke(fake_project_cli, ['catalog', 'create', '--pipeline', self.PIPELINE_NAME], obj=fake_metadata)
        assert not result.exit_code
        expected_catalog_config: Dict[str, Dict[str, str]] = {'example_test_x': catalog_config['example_test_x'], 'example_test_y': {'type': 'MemoryDataset'}, 'example_train_x': {'type': 'MemoryDataset'}, 'example_train_y': {'type': 'MemoryDataset'}}
        catalog_config = yaml.safe_load(data_catalog_file.read_text())
        assert catalog_config == expected_catalog_config

    def test_bad_env(self, fake_project_cli: Any, fake_metadata: Any) -> None:
        """Test error when provided conf environment does not exist"""
        env = 'no_such_env'
        cmd = ['catalog', 'list', '-e', env, '--pipeline', PIPELINE_NAME]
        result = CliRunner().invoke(fake_project_cli, cmd, obj=fake_metadata)
        assert result.exit_code
        assert 'Unable to instantiate Kedro Catalog' in result.output

@pytest.mark.usefixtures('chdir_to_dummy_project', 'fake_load_context')
class TestCatalogFactoryCommands:

    @pytest.mark.parametrize('catalog_type', [DataCatalog, KedroDataCatalog])
    @pytest.mark.usefixtures('mock_pipelines')
    def test_rank_catalog_factories(self, fake_project_cli: Any, fake_metadata: Any, mocker: Any, fake_load_context: MagicMock, fake_catalog_with_overlapping_factories: Dict[str, Dict[str, Any]], catalog_type: Any) -> None:
        yaml_dump_mock = mocker.patch('yaml.dump', return_value='Result YAML')
        mocked_context = fake_load_context.return_value
        mocked_context.catalog = catalog_type.from_config(fake_catalog_with_overlapping_factories)
        result = CliRunner().invoke(fake_project_cli, ['catalog', 'rank'], obj=fake_metadata)
        assert not result.exit_code
        expected_patterns_sorted: List[str] = ['an_example_{place}_{holder}', 'an_example_{placeholder}', 'an_{example_placeholder}', 'on_{example_placeholder}']
        assert yaml_dump_mock.call_count == 1
        assert yaml_dump_mock.call_args[0][0] == expected_patterns_sorted

    @pytest.mark.parametrize('catalog_type', [DataCatalog, KedroDataCatalog])
    def test_rank_catalog_factories_with_no_factories(self, fake_project_cli: Any, fake_metadata: Any, fake_load_context: MagicMock, catalog_type: Any) -> None:
        mocked_context = fake_load_context.return_value
        catalog_datasets: Dict[str, Any] = {'iris_data': CSVDataset(filepath='test.csv'), 'intermediate': MemoryDataset(), 'not_used': CSVDataset(filepath='test2.csv')}
        mocked_context.catalog = catalog_type(datasets=catalog_datasets)
        result = CliRunner().invoke(fake_project_cli, ['catalog', 'rank'], obj=fake_metadata)
        assert not result.exit_code
        expected_output = 'There are no dataset factories in the catalog.'
        assert expected_output in result.output

    @pytest.mark.parametrize('catalog_type', [DataCatalog, KedroDataCatalog])
    @pytest.mark.usefixtures('mock_pipelines')
    def test_catalog_resolve(self, fake_project_cli: Any, fake_metadata: Any, fake_load_context: MagicMock, mocker: Any, mock_pipelines: Dict[str, Any], fake_catalog_config: Dict[str, Dict[str, Any]], fake_catalog_config_resolved: Dict[str, Dict[str, Any]], fake_credentials_config: Dict[str, Dict[str, str]], catalog_type: Any) -> None:
        """Test that datasets factories are correctly resolved to the explicit datasets in the pipeline."""
        mocked_context = fake_load_context.return_value
        mocked_context.config_loader = {'catalog': fake_catalog_config, 'credentials': fake_credentials_config}
        mocked_context._get_config_credentials.return_value = fake_credentials_config
        mocked_context.catalog = DataCatalog.from_config(catalog=fake_catalog_config, credentials=fake_credentials_config)
        placeholder_ds = mocked_context.catalog.config_resolver.list_patterns()
        pipeline_datasets: Set[str] = {'csv_example', 'parquet_example', 'explicit_dataset'}
        mocker.patch.object(mock_pipelines[PIPELINE_NAME], 'datasets', return_value=pipeline_datasets)
        result = CliRunner().invoke(fake_project_cli, ['catalog', 'resolve'], obj=fake_metadata)
        assert not result.exit_code
        resolved_config = yaml.safe_load(result.output)
        assert resolved_config == fake_catalog_config_resolved
        for ds in placeholder_ds:
            assert ds not in result.output

    @pytest.mark.parametrize('catalog_type', [DataCatalog, KedroDataCatalog])
    @pytest.mark.usefixtures('mock_pipelines')
    def test_catalog_resolve_nested_config(self, fake_project_cli: Any, fake_metadata: Any, fake_load_context: MagicMock, mocker: Any, mock_pipelines: Dict[str, Any], fake_catalog_config_with_factories: Dict[str, Dict[str, Any]], fake_catalog_config_with_factories_resolved: Dict[str, Dict[str, Any]], catalog_type: Any) -> None:
        """Test that explicit catalog entries are not overwritten by factory config."""
        mocked_context = fake_load_context.return_value
        mocked_context.project_path = fake_metadata.project_path
        mocked_context.config_loader = {'catalog': fake_catalog_config_with_factories}
        mocked_context.catalog = catalog_type.from_config(fake_catalog_config_with_factories)
        mocker.patch.object(mock_pipelines[PIPELINE_NAME], 'datasets', return_value=mocked_context.catalog._datasets.keys() | {'csv_example', 'parquet_example', 'partitioned_example'})
        result = CliRunner().invoke(fake_project_cli, ['catalog', 'resolve'], obj=fake_metadata)
        assert not result.exit_code
        resolved_config = yaml.safe_load(result.output)
        assert resolved_config == fake_catalog_config_with_factories_resolved

    @pytest.mark.parametrize('catalog_type', [DataCatalog, KedroDataCatalog])
    @pytest.mark.usefixtures('mock_pipelines')
    def test_no_param_datasets_in_resolve(self, fake_project_cli: Any, fake_metadata: Any, fake_load_context: MagicMock, mocker: Any, mock_pipelines: Dict[str, Any], catalog_type: Any) -> None:
        yaml_dump_mock = mocker.patch('yaml.dump', return_value='Result YAML')
        mocked_context = fake_load_context.return_value
        catalog_config: Dict[str, Dict[str, Any]] = {'iris_data': {'type': 'pandas.CSVDataset', 'filepath': 'test.csv'}, 'intermediate': {'type': 'MemoryDataset'}}
        catalog_datasets: Dict[str, Any] = {'iris_data': CSVDataset(filepath='test.csv'), 'intermediate': MemoryDataset(), 'parameters': MemoryDataset(), 'params:data_ratio': MemoryDataset()}
        mocked_context.config_loader = {'catalog': catalog_config}
        mocked_context.catalog = catalog_type(datasets=catalog_datasets)
        mocker.patch.object(mock_pipelines[PIPELINE_NAME], 'datasets', return_value=catalog_datasets.keys())
        result = CliRunner().invoke(fake_project_cli, ['catalog', 'resolve'], obj=fake_metadata)
        assert not result.exit_code
        assert yaml_dump_mock.call_count == 1
        output = yaml_dump_mock.call_args[0][0]
        assert 'parameters' not in output.keys()
        assert 'params:data_ratio' not in output.keys()
        assert 'iris_data' in output.keys()
        assert 'intermediate' in output.keys()
