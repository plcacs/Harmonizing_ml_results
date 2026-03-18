```python
from argparse import Namespace
from typing import Any, Optional, Type, Union
from unittest.mock import MagicMock
import pytest
from psycopg2 import DatabaseError
from pytest_mock import MockerFixture
from dbt.adapters.contracts.connection import AdapterResponse
from dbt.adapters.postgres import PostgresAdapter
from dbt.artifacts.resources.v1.model import ModelConfig
from dbt.artifacts.schemas.results import RunStatus
from dbt.artifacts.schemas.run import RunResult
from dbt.config.runtime import RuntimeConfig
from dbt.contracts.graph.manifest import Manifest
from dbt.contracts.graph.nodes import HookNode, ModelNode
from dbt.exceptions import DbtRuntimeError
from dbt.flags import FlagDict
from dbt.task.run import MicrobatchModelRunner, ModelRunner, RunTask

def test_run_task_cancel_connections(
    exception_to_raise: Type[BaseException],
    expected_cancel_connections: bool,
    runtime_config: RuntimeConfig
) -> None: ...

def test_run_task_preserve_edges() -> None: ...

def test_tracking_fails_safely_for_missing_adapter() -> None: ...

def test_adapter_info_tracking() -> None: ...

class TestModelRunner:
    @pytest.fixture
    def log_model_result_catcher(self) -> Any: ...
    
    @pytest.fixture
    def model_runner(
        self,
        postgres_adapter: PostgresAdapter,
        table_model: ModelNode,
        runtime_config: RuntimeConfig
    ) -> ModelRunner: ...
    
    @pytest.fixture
    def run_result(self, table_model: ModelNode) -> RunResult: ...
    
    def test_print_result_line(
        self,
        log_model_result_catcher: Any,
        model_runner: ModelRunner,
        run_result: RunResult
    ) -> None: ...
    
    @pytest.mark.skip(reason="Default and adapter macros aren't being appropriately populated, leading to a runtime error")
    def test_execute(
        self,
        table_model: ModelNode,
        manifest: Manifest,
        model_runner: ModelRunner
    ) -> None: ...

class TestMicrobatchModelRunner:
    @pytest.fixture
    def model_runner(
        self,
        postgres_adapter: PostgresAdapter,
        table_model: ModelNode,
        runtime_config: RuntimeConfig
    ) -> MicrobatchModelRunner: ...
    
    @pytest.mark.parametrize('has_relation,relation_type,materialized,full_refresh_config,full_refresh_flag,expectation', [(False, 'table', 'incremental', None, False, False), (True, 'other', 'incremental', None, False, False), (True, 'table', 'other', None, False, False), (True, 'table', 'incremental', True, False, False), (True, 'table', 'incremental', True, True, False), (True, 'table', 'incremental', False, False, True), (True, 'table', 'incremental', False, True, True), (True, 'table', 'incremental', None, True, False), (True, 'table', 'incremental', None, False, True)])
    def test__is_incremental(
        self,
        mocker: MockerFixture,
        model_runner: MicrobatchModelRunner,
        has_relation: bool,
        relation_type: str,
        materialized: str,
        full_refresh_config: Optional[bool],
        full_refresh_flag: bool,
        expectation: bool
    ) -> None: ...
    
    @pytest.mark.parametrize('adapter_microbatch_concurrency,has_relation,concurrent_batches,has_this,expectation', [(True, True, None, False, True), (True, True, None, True, False), (True, True, True, False, True), (True, True, True, True, True), (True, True, False, False, False), (True, True, False, True, False), (True, False, None, False, False), (True, False, None, True, False), (True, False, True, False, False), (True, False, True, True, False), (True, False, False, False, False), (True, False, False, True, False), (False, True, None, False, False), (False, True, None, True, False), (False, True, True, False, False), (False, True, True, True, False), (False, True, False, False, False), (False, True, False, True, False), (False, False, None, False, False), (False, False, None, True, False), (False, False, True, False, False), (False, False, True, True, False), (False, False, False, False, False), (False, False, False, True, False)])
    def test_should_run_in_parallel(
        self,
        mocker: MockerFixture,
        model_runner: MicrobatchModelRunner,
        adapter_microbatch_concurrency: bool,
        has_relation: bool,
        concurrent_batches: Optional[bool],
        has_this: bool,
        expectation: bool
    ) -> None: ...

class TestRunTask:
    @pytest.fixture
    def hook_node(self) -> HookNode: ...
    
    @pytest.mark.parametrize('error_to_raise,expected_result', [(None, RunStatus.Success), (DbtRuntimeError, RunStatus.Error), (DatabaseError, RunStatus.Error), (KeyboardInterrupt, KeyboardInterrupt)])
    def test_safe_run_hooks(
        self,
        mocker: MockerFixture,
        runtime_config: RuntimeConfig,
        manifest: Manifest,
        hook_node: HookNode,
        error_to_raise: Optional[Type[BaseException]],
        expected_result: Union[RunStatus, Type[BaseException]]
    ) -> None: ...
```