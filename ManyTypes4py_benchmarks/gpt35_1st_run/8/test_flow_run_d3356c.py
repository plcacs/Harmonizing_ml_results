from typing import Any, Awaitable, Callable, List, Tuple, Union
from uuid import UUID, uuid4
import pytest
import prefect.exceptions
from prefect import flow
from prefect.cli.flow_run import LOGS_WITH_LIMIT_FLAG_DEFAULT_NUM_LOGS
from prefect.client.orchestration import PrefectClient, SyncPrefectClient
from prefect.client.schemas.actions import LogCreate
from prefect.client.schemas.objects import FlowRun
from prefect.deployments.runner import RunnerDeployment
from prefect.states import AwaitingRetry, Cancelled, Completed, Crashed, Failed, Late, Pending, Retrying, Running, Scheduled, State, StateType
from prefect.testing.cli import invoke_and_assert
from prefect.types import DateTime
from prefect.utilities.asyncutils import run_sync_in_worker_thread

def hello_flow() -> str:
    return 'Hello!'

def goodbye_flow() -> str:
    return 'Goodbye'

async def assert_flow_run_is_deleted(prefect_client: PrefectClient, flow_run_id: UUID) -> None:
    ...

def assert_flow_run_is_deleted_sync(prefect_client: PrefectClient, flow_run_id: UUID) -> None:
    ...

def assert_flow_runs_in_result(result: Any, expected: List[FlowRun], unexpected: List[FlowRun] = None) -> None:
    ...

async def scheduled_flow_run(prefect_client: PrefectClient) -> FlowRun:
    ...

async def completed_flow_run(prefect_client: PrefectClient) -> FlowRun:
    ...

async def running_flow_run(prefect_client: PrefectClient) -> FlowRun:
    ...

async def late_flow_run(prefect_client: PrefectClient) -> FlowRun:
    ...

def test_delete_flow_run_fails_correctly() -> None:
    ...

def test_delete_flow_run_succeeds(sync_prefect_client: SyncPrefectClient, flow_run: FlowRun) -> None:
    ...

def test_ls_no_args(scheduled_flow_run: FlowRun, completed_flow_run: FlowRun, running_flow_run: FlowRun, late_flow_run: FlowRun) -> None:
    ...

def test_ls_flow_name_filter(scheduled_flow_run: FlowRun, completed_flow_run: FlowRun, running_flow_run: FlowRun, late_flow_run: FlowRun) -> None:
    ...

def test_ls_state_type_filter_invalid_raises() -> None:
    ...

def test_ls_state_name_filter_unofficial_state_warns(caplog: Any) -> None:
    ...

def test_ls_limit(scheduled_flow_run: FlowRun, completed_flow_run: FlowRun, running_flow_run: FlowRun, late_flow_run: FlowRun) -> None:
    ...

class TestCancelFlowRun:

    async def test_non_terminal_states_set_to_cancelling(self, prefect_client: PrefectClient, state: State) -> None:
        ...

    async def test_scheduled_states_set_to_cancelled(self, prefect_client: PrefectClient, state: State) -> None:
        ...

    async def test_cancelling_terminal_states_exits_with_error(self, prefect_client: PrefectClient, state: State) -> None:
        ...

    def test_wrong_id_exits_with_error(self) -> None:
        ...

@pytest.fixture()
def flow_run_factory(prefect_client: PrefectClient) -> Callable[[int], Awaitable[FlowRun]]:
    ...

class TestFlowRunLogs:

    async def test_when_num_logs_smaller_than_page_size_then_no_pagination(self, flow_run_factory: Callable[[int], Awaitable[FlowRun]]) -> None:
        ...

    async def test_when_num_logs_greater_than_page_size_then_pagination(self, flow_run_factory: Callable[[int], Awaitable[FlowRun]]) -> None:
        ...

    async def test_when_flow_run_not_found_then_exit_with_error(self, flow_run_factory: Callable[[int], Awaitable[FlowRun]]) -> None:
        ...

    async def test_when_num_logs_smaller_than_page_size_with_head_then_no_pagination(self, flow_run_factory: Callable[[int], Awaitable[FlowRun]]) -> None:
        ...

    async def test_when_num_logs_greater_than_page_size_with_head_then_pagination(self, flow_run_factory: Callable[[int], Awaitable[FlowRun]]) -> None:
        ...

    async def test_when_num_logs_greater_than_page_size_with_head_outputs_correct_num_logs(self, flow_run_factory: Callable[[int], Awaitable[FlowRun]]) -> None:
        ...

    async def test_default_head_returns_default_num_logs(self, flow_run_factory: Callable[[int], Awaitable[FlowRun]]) -> None:
        ...

    async def test_h_and_n_shortcuts_for_head_and_num_logs(self, flow_run_factory: Callable[[int], Awaitable[FlowRun]]) -> None:
        ...

    async def test_num_logs_passed_standalone_returns_num_logs(self, flow_run_factory: Callable[[int], Awaitable[FlowRun]]) -> None:
        ...

    async def test_when_num_logs_passed_with_reverse_param_and_num_logs(self, flow_run_factory: Callable[[int], Awaitable[FlowRun]]) -> None:
        ...

    async def test_passing_head_and_tail_raises(self, flow_run_factory: Callable[[int], Awaitable[FlowRun]]) -> None:
        ...

    async def test_default_tail_returns_default_num_logs(self, flow_run_factory: Callable[[int], Awaitable[FlowRun]]) -> None:
        ...

    async def test_reverse_tail_with_num_logs(self, flow_run_factory: Callable[[int], Awaitable[FlowRun]]) -> None:
        ...

    async def test_reverse_tail_returns_default_num_logs(self, flow_run_factory: Callable[[int], Awaitable[FlowRun]]) -> None:
        ...

    async def test_when_num_logs_greater_than_page_size_with_tail_outputs_correct_num_logs(self, flow_run_factory: Callable[[int], Awaitable[FlowRun]]) -> None:
        ...

class TestFlowRunExecute:

    async def test_execute_flow_run_via_argument(self, prefect_client: PrefectClient) -> None:
        ...

    async def test_execute_flow_run_via_environment_variable(self, prefect_client: PrefectClient, monkeypatch: Any) -> None:
        ...
