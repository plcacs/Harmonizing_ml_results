from __future__ import annotations
import logging
from logging.handlers import QueueHandler, QueueListener
from multiprocessing import Queue
from typing import TYPE_CHECKING, Any
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
MOCK_PACKAGE_NAME = 'fake_package'

@pytest.fixture
def mock_package_name() -> Union[str, list]:
    return MOCK_PACKAGE_NAME

def _write_yaml(filepath: pathlib.Path, config: Union[dict, typing.MutableMapping, str]) -> None:
    filepath.parent.mkdir(parents=True, exist_ok=True)
    yaml_str = yaml.dump(config)
    filepath.write_text(yaml_str)

def _write_toml(filepath: pathlib.Path, config: Union[dict, typing.MutableMapping]) -> None:
    filepath.parent.mkdir(parents=True, exist_ok=True)
    toml_str = toml.dumps(config)
    filepath.write_text(toml_str)

def _assert_hook_call_record_has_expected_parameters(call_record: Union[logging.LogRecord, dict[str, typing.Any]], expected_parameters: Union[list[str], dict[str, typing.Any], dict]) -> None:
    """Assert the given call record has all expected parameters."""
    for param in expected_parameters:
        assert hasattr(call_record, param)

def _assert_pipeline_equal(p: Union[typing.Iterable[typing.Any], typing.AbstractSet], q: Union[typing.Iterable[typing.Any], typing.AbstractSet]) -> None:
    assert sorted(p.nodes) == sorted(q.nodes)

@pytest.fixture
def local_config(tmp_path: pathlib.Path) -> dict[typing.Text, dict[typing.Text, typing.Union[typing.Text,dict[typing.Text, bool],bool]]]:
    cars_filepath = str(tmp_path / 'cars.csv')
    boats_filepath = str(tmp_path / 'boats.csv')
    return {'cars': {'type': 'pandas.CSVDataset', 'filepath': cars_filepath, 'save_args': {'index': False}, 'versioned': True}, 'boats': {'type': 'pandas.CSVDataset', 'filepath': boats_filepath, 'versioned': True}}

@pytest.fixture(autouse=True)
def config_dir(tmp_path: Union[pathlib.Path, pathlib.PurePath], local_config: str) -> None:
    catalog = tmp_path / 'conf' / 'base' / 'catalog.yml'
    credentials = tmp_path / 'conf' / 'local' / 'credentials.yml'
    pyproject_toml = tmp_path / 'pyproject.toml'
    _write_yaml(catalog, local_config)
    _write_yaml(credentials, {'dev_s3': 'foo'})
    payload = {'tool': {'kedro': {'kedro_init_version': kedro_version, 'project_name': 'test hooks', 'package_name': 'test_hooks'}}}
    _write_toml(pyproject_toml, payload)

def identity_node(x: Union[T, str, bytes]) -> Union[T, str, bytes]:
    return x

def assert_exceptions_equal(e1: Union[Exception, BaseException, str], e2: Union[Exception, BaseException, str]) -> None:
    assert isinstance(e1, type(e2)) and str(e1) == str(e2)

@pytest.fixture
def dummy_dataframe():
    return pd.DataFrame({'test': [1, 2]})

@pytest.fixture
def mock_pipeline() -> Union[typing.Callable[T, typing.Union[T,None]], bool, typing.Iterable[str]]:
    return modular_pipeline([node(identity_node, 'cars', 'planes', name='node1'), node(identity_node, 'boats', 'ships', name='node2')], tags='pipeline')

class LogRecorder(logging.Handler):
    """Record logs received from a process-safe log listener"""

    def __init__(self) -> None:
        super().__init__()
        self.log_records = []

    def handle(self, record: Union[list[dict], dict]) -> None:
        self.log_records.append(record)

class LogsListener(QueueListener):
    """Listen to logs stream and capture log records with LogRecorder."""

    def __init__(self) -> None:
        queue = Queue()
        self.log_handler = QueueHandler(queue)
        logger.addHandler(self.log_handler)
        self.log_recorder = LogRecorder()
        super().__init__(queue, self.log_recorder)

    @property
    def logs(self):
        return self.log_recorder.log_records

@pytest.fixture
def logs_listener() -> typing.Generator[LogsListener]:
    """Fixture to start the logs listener before a test and clean up after the test finishes"""
    listener = LogsListener()
    listener.start()
    yield listener
    logger.removeHandler(listener.log_handler)
    listener.stop()

class LoggingHooks:
    """A set of test hooks that only log information when invoked"""

    @hook_impl
    def after_catalog_created(self, catalog: Union[dict[str, typing.Any], str, dict[str, str]], conf_catalog: Union[dict[str, typing.Any], str, dict[str, str]], conf_creds: Union[dict[str, typing.Any], str, dict[str, str]], feed_dict: Union[dict[str, typing.Any], str, dict[str, str]], save_version: Union[dict[str, typing.Any], str, dict[str, str]], load_versions: Union[dict[str, typing.Any], str, dict[str, str]]) -> None:
        logger.info('Catalog created', extra={'catalog': catalog, 'conf_catalog': conf_catalog, 'conf_creds': conf_creds, 'feed_dict': feed_dict, 'save_version': save_version, 'load_versions': load_versions})

    @hook_impl
    def before_node_run(self, node: Union[typing.Any, None, list[str], dict], catalog: Union[typing.Any, None, list[str], dict], inputs: Union[typing.Any, None, list[str], dict], is_async: Union[typing.Any, None, list[str], dict], session_id: Union[typing.Any, None, list[str], dict]) -> None:
        logger.info('About to run node', extra={'node': node, 'catalog': catalog, 'inputs': inputs, 'is_async': is_async, 'session_id': session_id})

    @hook_impl
    def after_node_run(self, node: Union[dict[str, typing.Any], str, kedro.io.DataCatalog], catalog: Union[dict[str, typing.Any], str, kedro.io.DataCatalog], inputs: Union[dict[str, typing.Any], str, kedro.io.DataCatalog], outputs: Union[dict[str, typing.Any], str, kedro.io.DataCatalog], is_async: Union[dict[str, typing.Any], str, kedro.io.DataCatalog], session_id: Union[dict[str, typing.Any], str, kedro.io.DataCatalog]) -> None:
        logger.info('Ran node', extra={'node': node, 'catalog': catalog, 'inputs': inputs, 'outputs': outputs, 'is_async': is_async, 'session_id': session_id})

    @hook_impl
    def on_node_error(self, error: Union[Exception, str, bool], node: Union[Exception, str, bool], catalog: Union[Exception, str, bool], inputs: Union[Exception, str, bool], is_async: Union[Exception, str, bool], session_id: Union[Exception, str, bool]) -> None:
        logger.info('Node error', extra={'error': error, 'node': node, 'catalog': catalog, 'inputs': inputs, 'is_async': is_async, 'session_id': session_id})

    @hook_impl
    def before_pipeline_run(self, run_params: Union[kedro.pipeline.Pipeline, dict[str, typing.Any], kedro.io.DataCatalog], pipeline: Union[kedro.pipeline.Pipeline, dict[str, typing.Any], kedro.io.DataCatalog], catalog: Union[kedro.pipeline.Pipeline, dict[str, typing.Any], kedro.io.DataCatalog]) -> None:
        logger.info('About to run pipeline', extra={'pipeline': pipeline, 'run_params': run_params, 'catalog': catalog})

    @hook_impl
    def after_pipeline_run(self, run_params: Union[Model, str, dict], run_result: Union[Model, str, dict], pipeline: Union[Model, str, dict], catalog: Union[Model, str, dict]) -> None:
        logger.info('Ran pipeline', extra={'pipeline': pipeline, 'run_params': run_params, 'run_result': run_result, 'catalog': catalog})

    @hook_impl
    def on_pipeline_error(self, error: Union[kedro.pipeline.Pipeline, dict[str, typing.Any], kedro.io.DataCatalog], run_params: Union[kedro.pipeline.Pipeline, dict[str, typing.Any], kedro.io.DataCatalog], pipeline: Union[kedro.pipeline.Pipeline, dict[str, typing.Any], kedro.io.DataCatalog], catalog: Union[kedro.pipeline.Pipeline, dict[str, typing.Any], kedro.io.DataCatalog]) -> None:
        logger.info('Pipeline error', extra={'error': error, 'run_params': run_params, 'pipeline': pipeline, 'catalog': catalog})

    @hook_impl
    def before_dataset_loaded(self, dataset_name: str, node: str) -> None:
        logger.info('Before dataset loaded', extra={'dataset_name': dataset_name, 'node': node})

    @hook_impl
    def after_dataset_loaded(self, dataset_name: Union[str, typing.Any, None], data: Union[str, typing.Any, None], node: Union[str, typing.Any, None]) -> None:
        logger.info('After dataset loaded', extra={'dataset_name': dataset_name, 'data': data, 'node': node})

    @hook_impl
    def before_dataset_saved(self, dataset_name: Union[str, typing.Any, None], data: Union[str, typing.Any, None], node: Union[str, typing.Any, None]) -> None:
        logger.info('Before dataset saved', extra={'dataset_name': dataset_name, 'data': data, 'node': node})

    @hook_impl
    def after_dataset_saved(self, dataset_name: Union[str, typing.Any, None], data: Union[str, typing.Any, None], node: Union[str, typing.Any, None]) -> None:
        logger.info('After dataset saved', extra={'dataset_name': dataset_name, 'data': data, 'node': node})

    @hook_impl
    def after_context_created(self, context: Union[dict[str, str], str]) -> None:
        logger.info('After context created', extra={'context': context})

@pytest.fixture
def project_hooks() -> LoggingHooks:
    """A set of project hook implementations that log to stdout whenever it is invoked."""
    return LoggingHooks()

@pytest.fixture(autouse=True)
def mock_pipelines(mocker: Any, mock_pipeline: Any):

    def mock_register_pipelines() -> dict[typing.Text, ]:
        return {'__default__': mock_pipeline, 'pipe': mock_pipeline}
    mocker.patch.object(_ProjectPipelines, '_get_pipelines_registry_callable', return_value=mock_register_pipelines)
    return mock_register_pipelines()

def _mock_imported_settings_paths(mocker: pathlib.Path, mock_settings: pathlib.Path) -> pathlib.Path:
    for path in ['kedro.framework.session.session.settings', 'kedro.framework.project.settings', 'kedro.runner.task.settings']:
        mocker.patch(path, mock_settings)
    return mock_settings

@pytest.fixture
def mock_settings(mocker: Union[str, kata.data.io.network.GithubApi], project_hooks: Union[bool, str, typing.Type]) -> pathlib.Path:

    class MockSettings(_ProjectSettings):
        _HOOKS = Validator('HOOKS', default=(project_hooks,))
    return _mock_imported_settings_paths(mocker, MockSettings())

@pytest.fixture
def mock_session(mock_settings: Union[dict, str, bool, None], mock_package_name: Union[str, path.Path, dict[str, typing.Any]], tmp_path: Union[str, audiopyle.lib.db.session.SessionProvider, dict]) -> typing.Generator:
    configure_project(mock_package_name)
    session = KedroSession.create(tmp_path, extra_params={'params:key': 'value'})
    yield session
    session.close()

@pytest.fixture(autouse=True)
def mock_validate_settings(mocker: Union[list[str], typing.Callable, str]) -> None:
    mocker.patch('kedro.framework.session.session.validate_settings')