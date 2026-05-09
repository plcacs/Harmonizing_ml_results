import logging
import multiprocessing
import re
import time
from typing import Any, Optional
import pandas as pd
import pytest
from dynaconf.validator import Validator
from kedro.framework.context.context import _convert_paths_to_absolute_posix
from kedro.framework.hooks import _create_hook_manager, hook_impl
from kedro.framework.hooks.manager import _register_hooks, _register_hooks_entry_points
from kedro.framework.project import _ProjectPipelines, _ProjectSettings, pipelines, settings
from kedro.framework.session import KedroSession
from kedro.io import DataCatalog, MemoryDataset
from kedro.pipeline import node, pipeline
from kedro.pipeline.node import Node
from kedro.runner import ParallelRunner
from kedro.runner.task import Task
from tests.framework.session.conftest import _assert_hook_call_record_has_expected_parameters, _assert_pipeline_equal, _mock_imported_settings_paths, assert_exceptions_equal
SKIP_ON_WINDOWS_AND_MACOS = pytest.mark.skipif(multiprocessing.get_start_method() == 'spawn', reason='Due to bug in parallel runner')
logger: logging.Logger = logging.getLogger('tests.framework.session.conftest')
logger.setLevel(logging.DEBUG)

def broken_node() -> Node:
    raise ValueError('broken')

@pytest.fixture
def broken_pipeline() -> pipeline:
    return pipeline([node(broken_node, None, 'A', name='node1'), node(broken_node, None, 'B', name='node2')], tags='pipeline')

@pytest.fixture
def mock_broken_pipelines(mocker: pytest.Mocker, broken_pipeline: pipeline) -> callable:
    def mock_get_pipelines_registry_callable() -> dict:
        return {'__default__': broken_pipeline}
    mocker.patch.object(_ProjectPipelines, '_get_pipelines_registry_callable', return_value=mock_get_pipelines_registry_callable)
    return mock_get_pipelines_registry_callable

class TestCatalogHooks:
    # ...

class TestPipelineHooks:
    # ...

class TestNodeHooks:
    # ...

class TestDatasetHooks:
    # ...

class MockDatasetReplacement:
    pass

@pytest.fixture
def mock_session_with_before_node_run_hooks(mocker: pytest.Mocker, project_hooks: Any, mock_package_name: str, tmp_path: str) -> KedroSession:
    # ...

@pytest.fixture
def mock_session_with_broken_before_node_run_hooks(mocker: pytest.Mocker, project_hooks: Any, mock_package_name: str, tmp_path: str) -> KedroSession:
    # ...

class TestAsyncNodeDatasetHooks:
    # ...

class TestKedroContextSpecsHook:
    # ...
