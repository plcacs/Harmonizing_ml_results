from __future__ import annotations
import logging
from logging.handlers import QueueHandler, QueueListener
from multiprocessing import Queue
from typing import TYPE_CHECKING, Any, Dict, List, Callable, Optional, Union, Set, Tuple
import pandas as pd
import pytest
import toml
import yaml
from dynaconf.validator import Validator
from kedro import __version__ as kedro_version
from kedro.framework.hooks import hook_impl
from kedro.framework.project import _ProjectPipelines, _ProjectSettings, configure_project
from kedro.framework.session import KedroSession
from kedro.pipeline.modular_pipeline import pipeline as modular_pipeline
from kedro.pipeline.node import Node, node
if TYPE_CHECKING:
    from pathlib import Path
    from kedro.framework.context.context import KedroContext
    from kedro.io import DataCatalog
    from kedro.pipeline import Pipeline
logger = logging.getLogger(__name__)
MOCK_PACKAGE_NAME: str = 'fake_package'

@pytest.fixture
def mock_package_name() -> str:
    return MOCK_PACKAGE_NAME

def _write_yaml(filepath: Path, config: Dict[str, Any]) -> None:
    filepath.parent.mkdir(parents=True, exist_ok=True)
    yaml_str = yaml.dump(config)
    filepath.write_text(yaml_str)

def _write_toml(filepath: Path, config: Dict[str, Any]) -> None:
    filepath.parent.mkdir(parents=True, exist_ok=True)
    toml_str = toml.dumps(config)
    filepath.write_text(toml_str)

def _assert_hook_call_record_has_expected_parameters(call_record: Any, expected_parameters: List[str]) -> None:
    """Assert the given call record has all expected parameters."""
    for param in expected_parameters:
        assert hasattr(call_record, param)

def _assert_pipeline_equal(p: Pipeline, q: Pipeline) -> None:
    assert sorted(p.nodes) == sorted(q.nodes)

@pytest.fixture
def local_config(tmp_path: Path) -> Dict[str, Dict[str, Any]]:
    cars_filepath = str(tmp_path / 'cars.csv')
    boats_filepath = str(tmp_path / 'boats.csv')
    return {'cars': {'type': 'pandas.CSVDataset', 'filepath': cars_filepath, 'save_args': {'index': False}, 'versioned': True}, 'boats': {'type': 'pandas.CSVDataset', 'filepath': boats_filepath, 'versioned': True}}

@pytest.fixture(autouse=True)
def config_dir(tmp_path: Path, local_config: Dict[str, Dict[str, Any]]) -> None:
    catalog = tmp_path / 'conf' / 'base' / 'catalog.yml'
    credentials = tmp_path / 'conf' / 'local' / 'credentials.yml'
    pyproject_toml = tmp_path / 'pyproject.toml'
    _write_yaml(catalog, local_config)
    _write_yaml(credentials, {'dev_s3': 'foo'})
    payload = {'tool': {'kedro': {'kedro_init_version': kedro_version, 'project_name': 'test hooks', 'package_name': 'test_hooks'}}}
    _write_toml(pyproject_toml, payload)

def identity_node(x: Any) -> Any:
    return x

def assert_exceptions_equal(e1: Exception, e2: Exception) -> None:
    assert isinstance(e1, type(e2)) and str(e1) == str(e2)

@pytest.fixture
def dummy_dataframe() -> pd.DataFrame:
    return pd.DataFrame({'test': [1, 2]})

@pytest.fixture
def mock_pipeline() -> Pipeline:
    return modular_pipeline([node(identity_node, 'cars', 'planes', name='node1'), node(identity_node, 'boats', 'ships', name='node2')], tags='pipeline')

class LogRecorder(logging.Handler):
    """Record logs received from a process-safe log listener"""

    def __init__(self) -> None:
        super().__init__()
        self.log_records: List[logging.LogRecord] = []

    def handle(self, record: logging.LogRecord) -> None:
        self.log_records.append(record)

class LogsListener(QueueListener):
    """Listen to logs stream and capture log records with LogRecorder."""

    def __init__(self) -> None:
        queue: Queue = Queue()
        self.log_handler: QueueHandler = QueueHandler(queue)
        logger.addHandler(self.log_handler)
        self.log_recorder: LogRecorder = LogRecorder()
        super().__init__(queue, self.log_recorder)

    @property
    def logs(self) -> List[logging.LogRecord]:
        return self.log_recorder.log_records

@pytest.fixture
def logs_listener() -> LogsListener:
    """Fixture to start the logs listener before a test and clean up after the test finishes"""
    listener = LogsListener()
    listener.start()
    yield listener
    logger.removeHandler(listener.log_handler)
    listener.stop()

class LoggingHooks:
    """A set of test hooks that only log information when invoked"""

    @hook_impl
    def after_catalog_created(self, catalog: DataCatalog, conf_catalog: Dict[str, Any], conf_creds: Dict[str, Any], feed_dict: Dict[str, Any], save_version: str, load_versions: Dict[str, str]) -> None:
        logger.info('Catalog created', extra={'catalog': catalog, 'conf_catalog': conf_catalog, 'conf_creds': conf_creds, 'feed_dict': feed_dict, 'save_version': save_version, 'load_versions': load_versions})

    @hook_impl
    def before_node_run(self, node: Node, catalog: DataCatalog, inputs: Dict[str, Any], is_async: bool, session_id: str) -> None:
        logger.info('About to run node', extra={'node': node, 'catalog': catalog, 'inputs': inputs, 'is_async': is_async, 'session_id': session_id})

    @hook_impl
    def after_node_run(self, node: Node, catalog: DataCatalog, inputs: Dict[str, Any], outputs: Dict[str, Any], is_async: bool, session_id: str) -> None:
        logger.info('Ran node', extra={'node': node, 'catalog': catalog, 'inputs': inputs, 'outputs': outputs, 'is_async': is_async, 'session_id': session_id})

    @hook_impl
    def on_node_error(self, error: Exception, node: Node, catalog: DataCatalog, inputs: Dict[str, Any], is_async: bool, session_id: str) -> None:
        logger.info('Node error', extra={'error': error, 'node': node, 'catalog': catalog, 'inputs': inputs, 'is_async': is_async, 'session_id': session_id})

    @hook_impl
    def before_pipeline_run(self, run_params: Dict[str, Any], pipeline: Pipeline, catalog: DataCatalog) -> None:
        logger.info('About to run pipeline', extra={'pipeline': pipeline, 'run_params': run_params, 'catalog': catalog})

    @hook_impl
    def after_pipeline_run(self, run_params: Dict[str, Any], run_result: Dict[str, Any], pipeline: Pipeline, catalog: DataCatalog) -> None:
        logger.info('Ran pipeline', extra={'pipeline': pipeline, 'run_params': run_params, 'run_result': run_result, 'catalog': catalog})

    @hook_impl
    def on_pipeline_error(self, error: Exception, run_params: Dict[str, Any], pipeline: Pipeline, catalog: DataCatalog) -> None:
        logger.info('Pipeline error', extra={'error': error, 'run_params': run_params, 'pipeline': pipeline, 'catalog': catalog})

    @hook_impl
    def before_dataset_loaded(self, dataset_name: str, node: Node) -> None:
        logger.info('Before dataset loaded', extra={'dataset_name': dataset_name, 'node': node})

    @hook_impl
    def after_dataset_loaded(self, dataset_name: str, data: Any, node: Node) -> None:
        logger.info('After dataset loaded', extra={'dataset_name': dataset_name, 'data': data, 'node': node})

    @hook_impl
    def before_dataset_saved(self, dataset_name: str, data: Any, node: Node) -> None:
        logger.info('Before dataset saved', extra={'dataset_name': dataset_name, 'data': data, 'node': node})

    @hook_impl
    def after_dataset_saved(self, dataset_name: str, data: Any, node: Node) -> None:
        logger.info('After dataset saved', extra={'dataset_name': dataset_name, 'data': data, 'node': node})

    @hook_impl
    def after_context_created(self, context: KedroContext) -> None:
        logger.info('After context created', extra={'context': context})

@pytest.fixture
def project_hooks() -> LoggingHooks:
    """A set of project hook implementations that log to stdout whenever it is invoked."""
    return LoggingHooks()

@pytest.fixture(autouse=True)
def mock_pipelines(mocker: Any, mock_pipeline: Pipeline) -> Dict[str, Pipeline]:

    def mock_register_pipelines() -> Dict[str, Pipeline]:
        return {'__default__': mock_pipeline, 'pipe': mock_pipeline}
    mocker.patch.object(_ProjectPipelines, '_get_pipelines_registry_callable', return_value=mock_register_pipelines)
    return mock_register_pipelines()

def _mock_imported_settings_paths(mocker: Any, mock_settings: _ProjectSettings) -> _ProjectSettings:
    for path in ['kedro.framework.session.session.settings', 'kedro.framework.project.settings', 'kedro.runner.task.settings']:
        mocker.patch(path, mock_settings)
    return mock_settings

@pytest.fixture
def mock_settings(mocker: Any, project_hooks: LoggingHooks) -> _ProjectSettings:

    class MockSettings(_ProjectSettings):
        _HOOKS = Validator('HOOKS', default=(project_hooks,))
    return _mock_imported_settings_paths(mocker, MockSettings())

@pytest.fixture
def mock_session(mock_settings: _ProjectSettings, mock_package_name: str, tmp_path: Path) -> KedroSession:
    configure_project(mock_package_name)
    session = KedroSession.create(tmp_path, extra_params={'params:key': 'value'})
    yield session
    session.close()

@pytest.fixture(autouse=True)
def mock_validate_settings(mocker: Any) -> None:
    mocker.patch('kedro.framework.session.session.validate_settings')
