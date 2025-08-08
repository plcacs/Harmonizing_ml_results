from __future__ import annotations
import logging
from logging.handlers import QueueHandler, QueueListener
from multiprocessing import Queue
from typing import TYPE_CHECKING, Any, Dict, List, Union
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
logger: logging.Logger = logging.getLogger(__name__)
MOCK_PACKAGE_NAME: str = 'fake_package'

def _write_yaml(filepath: Path, config: Dict[str, Any]) -> None:
    ...

def _write_toml(filepath: Path, config: Dict[str, Any]) -> None:
    ...

def _assert_hook_call_record_has_expected_parameters(call_record: Any, expected_parameters: List[str]) -> None:
    ...

def _assert_pipeline_equal(p: Pipeline, q: Pipeline) -> None:
    ...

@pytest.fixture
def local_config(tmp_path: Path) -> Dict[str, Any]:
    ...

@pytest.fixture(autouse=True)
def config_dir(tmp_path: Path, local_config: Dict[str, Any]) -> None:
    ...

def identity_node(x: Any) -> Any:
    ...

def assert_exceptions_equal(e1: Exception, e2: Exception) -> None:
    ...

@pytest.fixture
def dummy_dataframe() -> pd.DataFrame:
    ...

@pytest.fixture
def mock_pipeline() -> Pipeline:
    ...

class LogRecorder(logging.Handler):
    ...

class LogsListener(QueueListener):
    ...

@pytest.fixture
def logs_listener() -> LogsListener:
    ...

class LoggingHooks:
    ...

@pytest.fixture
def project_hooks() -> LoggingHooks:
    ...

@pytest.fixture(autouse=True)
def mock_pipelines(mocker, mock_pipeline: Pipeline) -> Dict[str, Pipeline]:
    ...

def _mock_imported_settings_paths(mocker, mock_settings: _ProjectSettings) -> _ProjectSettings:
    ...

@pytest.fixture
def mock_settings(mocker, project_hooks: LoggingHooks) -> _ProjectSettings:
    ...

@pytest.fixture
def mock_session(mock_settings: _ProjectSettings, mock_package_name: str, tmp_path: Path) -> KedroSession:
    ...

@pytest.fixture(autouse=True)
def mock_validate_settings(mocker) -> None:
    ...
