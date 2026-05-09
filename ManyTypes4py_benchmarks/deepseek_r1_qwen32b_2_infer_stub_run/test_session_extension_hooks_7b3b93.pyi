import logging
import multiprocessing
import re
import time
from typing import Any, Dict, List, Optional, Union
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
from tests.framework.session.conftest import _assert_hook_call_record_has_expected_parameters, _assert_pipeline_equal, _mock_imported_settings_paths, assert_exceptions_equal

SKIP_ON_WINDOWS_AND_MACOS = pytest.MarkDecorator
logger: logging.Logger

def broken_node() -> None:
    ...

@pytest.fixture
def broken_pipeline() -> pipeline:
    ...

@pytest.fixture
def mock_broken_pipelines(mocker, broken_pipeline) -> Dict[str, pipeline]:
    ...

class TestCatalogHooks:
    def test_after_catalog_created_hook(self, mock_session, caplog) -> None:
        ...

    def test_after_catalog_created_hook_on_session_run(self, mocker, mock_session, dummy_dataframe, caplog) -> None:
        ...

class TestPipelineHooks:
    @pytest.mark.usefixtures('mock_pipelines')
    def test_before_and_after_pipeline_run_hooks(self, caplog, mock_session, dummy_dataframe) -> None:
        ...

    @pytest.mark.usefixtures('mock_broken_pipelines')
    def test_on_pipeline_error_hook(self, caplog, mock_session) -> None:
        ...

    @pytest.mark.usefixtures('mock_broken_pipelines')
    def test_on_node_error_hook_sequential_runner(self, caplog, mock_session) -> None:
        ...

class TestNodeHooks:
    @pytest.mark.usefixtures('mock_pipelines')
    def test_before_and_after_node_run_hooks_sequential_runner(self, caplog, mock_session, dummy_dataframe) -> None:
        ...

    @SKIP_ON_WINDOWS_AND_MACOS
    @pytest.mark.usefixtures('mock_broken_pipelines')
    def test_on_node_error_hook_parallel_runner(self, mock_session, logs_listener) -> None:
        ...

    @SKIP_ON_WINDOWS_AND_MACOS
    @pytest.mark.usefixtures('mock_pipelines')
    def test_before_and_after_node_run_hooks_parallel_runner(self, mock_session, logs_listener, dummy_dataframe) -> None:
        ...

class TestDatasetHooks:
    @pytest.mark.usefixtures('mock_pipelines')
    def test_before_and_after_dataset_loaded_hooks_sequential_runner(self, mock_session, caplog, dummy_dataframe) -> None:
        ...

    @SKIP_ON_WINDOWS_AND_MACOS
    @pytest.mark.usefixtures('mock_settings')
    def test_before_and_after_dataset_loaded_hooks_parallel_runner(self, mock_session, logs_listener, dummy_dataframe) -> None:
        ...

    def test_before_and_after_dataset_saved_hooks_sequential_runner(self, mock_session, caplog, dummy_dataframe) -> None:
        ...

    @SKIP_ON_WINDOWS_AND_MACOS
    def test_before_and_after_dataset_saved_hooks_parallel_runner(self, mock_session, logs_listener, dummy_dataframe) -> None:
        ...

class TestBeforeNodeRunHookWithInputUpdates:
    def test_correct_input_update(self, mock_session_with_before_node_run_hooks, dummy_dataframe) -> None:
        ...

    @SKIP_ON_WINDOWS_AND_MACOS
    def test_correct_input_update_parallel(self, mock_session_with_before_node_run_hooks, dummy_dataframe) -> None:
        ...

    def test_broken_input_update(self, mock_session_with_broken_before_node_run_hooks, dummy_dataframe) -> None:
        ...

    @SKIP_ON_WINDOWS_AND_MACOS
    def test_broken_input_update_parallel(self, mock_session_with_broken_before_node_run_hooks, dummy_dataframe) -> None:
        ...

def wait_and_identity(*args: Any) -> Union[Any, tuple]:
    ...

@pytest.fixture
def sample_node() -> Node:
    ...

@pytest.fixture
def sample_node_multiple_outputs() -> Node:
    ...

class LogCatalog(DataCatalog):
    def load(self, name: str, version: Optional[str] = None) -> Any:
        ...

@pytest.fixture
def memory_catalog() -> LogCatalog:
    ...

@pytest.fixture
def hook_manager() -> _create_hook_manager:
    ...

class TestAsyncNodeDatasetHooks:
    @pytest.mark.usefixtures('mock_settings')
    def test_after_dataset_load_hook_async(self, memory_catalog, mock_session, sample_node, logs_listener) -> None:
        ...

    def test_after_dataset_load_hook_async_multiple_outputs(self, mocker, memory_catalog, hook_manager, sample_node_multiple_outputs) -> None:
        ...

class TestKedroContextSpecsHook:
    def test_after_context_created_hook(self, mock_session, caplog) -> None:
        ...