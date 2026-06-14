from argparse import Namespace
from typing import Optional, Type, Union
from unittest.mock import MagicMock

import pytest
from pytest_mock import MockerFixture

from dbt.adapters.postgres import PostgresAdapter
from dbt.artifacts.schemas.results import RunStatus
from dbt.artifacts.schemas.run import RunResult
from dbt.config.runtime import RuntimeConfig
from dbt.contracts.graph.manifest import Manifest
from dbt.contracts.graph.nodes import HookNode, ModelNode
from dbt.task.run import MicrobatchModelRunner, ModelRunner, RunTask
from tests.utils import EventCatcher


def test_run_task_cancel_connections(
    exception_to_raise: Type[BaseException],
    expected_cancel_connections: bool,
    runtime_config: RuntimeConfig,
) -> None: ...

def test_run_task_preserve_edges() -> None: ...

def test_tracking_fails_safely_for_missing_adapter() -> None: ...

def test_adapter_info_tracking() -> None: ...


class TestModelRunner:
    @pytest.fixture
    def log_model_result_catcher(self) -> EventCatcher: ...

    @pytest.fixture
    def model_runner(
        self,
        postgres_adapter: PostgresAdapter,
        table_model: ModelNode,
        runtime_config: RuntimeConfig,
    ) -> ModelRunner: ...

    @pytest.fixture
    def run_result(self, table_model: ModelNode) -> RunResult: ...

    def test_print_result_line(
        self,
        log_model_result_catcher: EventCatcher,
        model_runner: ModelRunner,
        run_result: RunResult,
    ) -> None: ...

    def test_execute(
        self,
        table_model: ModelNode,
        manifest: Manifest,
        model_runner: ModelRunner,
    ) -> None: ...


class TestMicrobatchModelRunner:
    @pytest.fixture
    def model_runner(
        self,
        postgres_adapter: PostgresAdapter,
        table_model: ModelNode,
        runtime_config: RuntimeConfig,
    ) -> MicrobatchModelRunner: ...

    def test__is_incremental(
        self,
        mocker: MockerFixture,
        model_runner: MicrobatchModelRunner,
        has_relation: bool,
        relation_type: str,
        materialized: str,
        full_refresh_config: Optional[bool],
        full_refresh_flag: bool,
        expectation: bool,
    ) -> None: ...

    def test_should_run_in_parallel(
        self,
        mocker: MockerFixture,
        model_runner: MicrobatchModelRunner,
        adapter_microbatch_concurrency: bool,
        has_relation: bool,
        concurrent_batches: Optional[bool],
        has_this: bool,
        expectation: bool,
    ) -> None: ...


class TestRunTask:
    @pytest.fixture
    def hook_node(self) -> HookNode: ...

    def test_safe_run_hooks(
        self,
        mocker: MockerFixture,
        runtime_config: RuntimeConfig,
        manifest: Manifest,
        hook_node: HookNode,
        error_to_raise: Optional[Type[BaseException]],
        expected_result: Union[RunStatus, Type[BaseException]],
    ) -> None: ...