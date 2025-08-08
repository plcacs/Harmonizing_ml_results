from argparse import Namespace
from dataclasses import dataclass
from importlib import import_module
from typing import Optional, Type, Union
from unittest import mock
from unittest.mock import MagicMock, patch
import pytest
from psycopg2 import DatabaseError
from pytest_mock import MockerFixture
from dbt.adapters.contracts.connection import AdapterResponse
from dbt.adapters.postgres import PostgresAdapter
from dbt.artifacts.resources.base import FileHash
from dbt.artifacts.resources.types import NodeType, RunHookType
from dbt.artifacts.resources.v1.components import DependsOn
from dbt.artifacts.resources.v1.config import NodeConfig
from dbt.artifacts.resources.v1.model import ModelConfig
from dbt.artifacts.schemas.results import RunStatus
from dbt.artifacts.schemas.run import RunResult
from dbt.config.runtime import RuntimeConfig
from dbt.contracts.graph.manifest import Manifest
from dbt.contracts.graph.nodes import HookNode, ModelNode
from dbt.events.types import LogModelResult
from dbt.exceptions import DbtRuntimeError
from dbt.flags import get_flags, set_from_args
from dbt.task.run import MicrobatchModelRunner, ModelRunner, RunTask, _get_adapter_info
from dbt.tests.util import safe_set_invocation_context
from dbt_common.events.base_types import EventLevel
from dbt_common.events.event_manager_client import add_callback_to_manager
from tests.utils import EventCatcher

@pytest.mark.parametrize('exception_to_raise: Type[BaseException], expected_cancel_connections: bool', [(SystemExit, True), (KeyboardInterrupt, True), (Exception, False)])
def test_run_task_cancel_connections(exception_to_raise: Type[BaseException], expected_cancel_connections: bool, runtime_config: RuntimeConfig) -> None:

def mock_run_queue(*args, **kwargs):
    raise exception_to_raise('Test exception')
with patch.object(RunTask, 'run_queue', mock_run_queue), patch.object(RunTask, '_cancel_connections') as mock_cancel_connections:
    set_from_args(Namespace(write_json=False), None)
    task = RunTask(get_flags(), runtime_config, None)
    with pytest.raises(exception_to_raise):
        task.execute_nodes()
    assert mock_cancel_connections.called == expected_cancel_connections

def test_run_task_preserve_edges() -> None:

def test_tracking_fails_safely_for_missing_adapter() -> None:

def test_adapter_info_tracking() -> None:

class TestModelRunner:

    @pytest.fixture
    def log_model_result_catcher(self) -> EventCatcher:

    @pytest.fixture
    def model_runner(self, postgres_adapter: PostgresAdapter, table_model: ModelNode, runtime_config: RuntimeConfig) -> ModelRunner:

    @pytest.fixture
    def run_result(self, table_model: ModelNode) -> RunResult:

    def test_print_result_line(self, log_model_result_catcher: EventCatcher, model_runner: ModelRunner, run_result: RunResult) -> None:

    @pytest.mark.skip(reason="Default and adapter macros aren't being appropriately populated, leading to a runtime error")
    def test_execute(self, table_model: ModelNode, manifest: Manifest, model_runner: ModelRunner) -> None:

class TestMicrobatchModelRunner:

    @pytest.fixture
    def model_runner(self, postgres_adapter: PostgresAdapter, table_model: ModelNode, runtime_config: RuntimeConfig) -> MicrobatchModelRunner:

    @pytest.mark.parametrize('has_relation: bool, relation_type: str, materialized: str, full_refresh_config: Optional[bool], full_refresh_flag: bool, expectation: bool', [(False, 'table', 'incremental', None, False, False), ...])
    def test__is_incremental(self, mocker, model_runner: MicrobatchModelRunner, has_relation: bool, relation_type: str, materialized: str, full_refresh_config: Optional[bool], full_refresh_flag: bool, expectation: bool) -> None:

    @pytest.mark.parametrize('adapter_microbatch_concurrency: bool, has_relation: bool, concurrent_batches: Optional[bool], has_this: bool, expectation: bool', [(True, True, None, False, True), ...])
    def test_should_run_in_parallel(self, mocker, model_runner: MicrobatchModelRunner, adapter_microbatch_concurrency: bool, has_relation: bool, concurrent_batches: Optional[bool], has_this: bool, expectation: bool) -> None:

class TestRunTask:

    @pytest.fixture
    def hook_node(self) -> HookNode:

    @pytest.mark.parametrize('error_to_raise: Optional[Type[BaseException]], expected_result: Union[RunStatus, Type[BaseException]]', [(None, RunStatus.Success), ...])
    def test_safe_run_hooks(self, mocker, runtime_config: RuntimeConfig, manifest: Manifest, hook_node: HookNode, error_to_raise: Optional[Type[BaseException]], expected_result: Union[RunStatus, Type[BaseException]]) -> None:
