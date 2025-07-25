from typing import Any, Awaitable, Callable, List, Optional, Sequence, Tuple, Type, Union
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

@flow(name='hello')
def hello_flow() -> str:
    return 'Hello!'

@flow(name='goodbye')
def goodbye_flow() -> str:
    return 'Goodbye'

async def assert_flow_run_is_deleted(prefect_client: PrefectClient, flow_run_id: UUID) -> None:
    """
    Make sure that the flow run created for our CLI test is actually deleted.
    """
    with pytest.raises(prefect.exceptions.ObjectNotFound):
        await prefect_client.read_flow_run(flow_run_id)

def assert_flow_run_is_deleted_sync(prefect_client: SyncPrefectClient, flow_run_id: UUID) -> None:
    """
    Make sure that the flow run created for our CLI test is actually deleted.
    """
    with pytest.raises(prefect.exceptions.ObjectNotFound):
        prefect_client.read_flow_run(flow_run_id)

def assert_flow_runs_in_result(result: Any, expected: List[FlowRun], unexpected: Optional[List[FlowRun]] = None) -> None:
    output = result.stdout.strip()
    for flow_run in expected:
        id_start = str(flow_run.id)[:20]
        assert id_start in output
    if unexpected:
        for flow_run in unexpected:
            id_start = str(flow_run.id)[:20]
            assert id_start not in output, f'{flow_run} should not be in the output'

@pytest.fixture
async def scheduled_flow_run(prefect_client: PrefectClient) -> FlowRun:
    return await prefect_client.create_flow_run(name='scheduled_flow_run', flow=hello_flow, state=Scheduled())

@pytest.fixture
async def completed_flow_run(prefect_client: PrefectClient) -> FlowRun:
    return await prefect_client.create_flow_run(name='completed_flow_run', flow=hello_flow, state=Completed())

@pytest.fixture
async def running_flow_run(prefect_client: PrefectClient) -> FlowRun:
    return await prefect_client.create_flow_run(name='running_flow_run', flow=goodbye_flow, state=Running())

@pytest.fixture
async def late_flow_run(prefect_client: PrefectClient) -> FlowRun:
    return await prefect_client.create_flow_run(name='late_flow_run', flow=goodbye_flow, state=Late())

def test_delete_flow_run_fails_correctly() -> None:
    missing_flow_run_id = 'ccb86ed0-e824-4d8b-b825-880401320e41'
    invoke_and_assert(command=['flow-run', 'delete', missing_flow_run_id], user_input='y', expected_output_contains=f"Flow run '{missing_flow_run_id}' not found!", expected_code=1)

def test_delete_flow_run_succeeds(sync_prefect_client: SyncPrefectClient, flow_run: FlowRun) -> None:
    invoke_and_assert(command=['flow-run', 'delete', str(flow_run.id)], user_input='y', expected_output_contains=f"Successfully deleted flow run '{str(flow_run.id)}'.", expected_code=0)
    assert_flow_run_is_deleted_sync(sync_prefect_client, flow_run.id)

def test_ls_no_args(scheduled_flow_run: FlowRun, completed_flow_run: FlowRun, running_flow_run: FlowRun, late_flow_run: FlowRun) -> None:
    result = invoke_and_assert(command=['flow-run', 'ls'], expected_code=0)
    assert_flow_runs_in_result(result, [scheduled_flow_run, completed_flow_run, running_flow_run, late_flow_run])

def test_ls_flow_name_filter(scheduled_flow_run: FlowRun, completed_flow_run: FlowRun, running_flow_run: FlowRun, late_flow_run: FlowRun) -> None:
    result = invoke_and_assert(command=['flow-run', 'ls', '--flow-name', 'goodbye'], expected_code=0)
    assert_flow_runs_in_result(result, expected=[running_flow_run, late_flow_run], unexpected=[scheduled_flow_run, completed_flow_run])

@pytest.mark.parametrize('state_type_1, state_type_2', [('completed', 'running'), ('COMPLETED', 'RUNNING'), ('Completed', 'Running')])
def test_ls_state_type_filter(scheduled_flow_run: FlowRun, completed_flow_run: FlowRun, running_flow_run: FlowRun, late_flow_run: FlowRun, state_type_1: str, state_type_2: str) -> None:
    result = invoke_and_assert(command=['flow-run', 'ls', '--state-type', state_type_1, '--state-type', state_type_2], expected_code=0)
    assert_flow_runs_in_result(result, expected=[running_flow_run, completed_flow_run], unexpected=[scheduled_flow_run, late_flow_run])

def test_ls_state_type_filter_invalid_raises() -> None:
    invoke_and_assert(command=['flow-run', 'ls', '--state-type', 'invalid'], expected_code=1, expected_output_contains='Invalid state type. Options are SCHEDULED, PENDING, RUNNING, COMPLETED, FAILED, CANCELLED, CRASHED, PAUSED, CANCELLING.')

@pytest.mark.parametrize('state_name', ['Late', 'LATE', 'late'])
def test_ls_state_name_filter(scheduled_flow_run: FlowRun, completed_flow_run: FlowRun, running_flow_run: FlowRun, late_flow_run: FlowRun, state_name: str) -> None:
    result = invoke_and_assert(command=['flow-run', 'ls', '--state', state_name], expected_code=0)
    assert_flow_runs_in_result(result, expected=[late_flow_run], unexpected=[running_flow_run, scheduled_flow_run, completed_flow_run])

def test_ls_state_name_filter_unofficial_state_warns(caplog: Any) -> None:
    invoke_and_assert(command=['flow-run', 'ls', '--state', 'MyCustomState'], expected_code=0, expected_output_contains=('No flow runs found.',))
    assert "State name 'MyCustomState' is not one of the official Prefect state names" in caplog.text

def test_ls_limit(scheduled_flow_run: FlowRun, completed_flow_run: FlowRun, running_flow_run: FlowRun, late_flow_run: FlowRun) -> None:
    result = invoke_and_assert(command=['flow-run', 'ls', '--limit', '2'], expected_code=0)
    output = result.stdout.strip()
    found_count = 0
    for flow_run in [scheduled_flow_run, completed_flow_run, running_flow_run, late_flow_run]:
        id_start = str(flow_run.id)[:20]
        if id_start in output:
            found_count += 1
    assert found_count == 2

class TestCancelFlowRun:

    @pytest.mark.parametrize('state', [Running, Pending, Retrying])
    async def test_non_terminal_states_set_to_cancelling(self, prefect_client: PrefectClient, state: Type[State]) -> None:
        """Should set the state of the flow to Cancelling. Does not include Scheduled
        states, because they should be set to Cancelled instead.
        """
        before = await prefect_client.create_flow_run(name='scheduled_flow_run', flow=hello_flow, state=state())
        await run_sync_in_worker_thread(invoke_and_assert, command=['flow-run', 'cancel', str(before.id)], expected_code=0, expected_output_contains=f"Flow run '{before.id}' was successfully scheduled for cancellation.")
        after = await prefect_client.read_flow_run(before.id)
        assert before.state.name != after.state.name
        assert before.state.type != after.state.type
        assert after.state.type == StateType.CANCELLING

    @pytest.mark.parametrize('state', [Scheduled, AwaitingRetry, Late])
    async def test_scheduled_states_set_to_cancelled(self, prefect_client: PrefectClient, state: Type[State]) -> None:
        """Should set the state of the flow run to Cancelled."""
        before = await prefect_client.create_flow_run(name='scheduled_flow_run', flow=hello_flow, state=state())
        await run_sync_in_worker_thread(invoke_and_assert, command=['flow-run', 'cancel', str(before.id)], expected_code=0, expected_output_contains=f"Flow run '{before.id}' was successfully scheduled for cancellation.")
        after = await prefect_client.read_flow_run(before.id)
        assert before.state.name != after.state.name
        assert before.state.type != after.state.type
        assert after.state.type == StateType.CANCELLED

    @pytest.mark.parametrize('state', [Completed, Failed, Crashed, Cancelled])
    async def test_cancelling_terminal_states_exits_with_error(self, prefect_client: PrefectClient, state: Type[State]) -> None:
        before = await prefect_client.create_flow_run(name='scheduled_flow_run', flow=hello_flow, state=state())
        await run_sync_in_worker_thread(invoke_and_assert, command=['flow-run', 'cancel', str(before.id)], expected_code=1, expected_output_contains=f"Flow run '{before.id}' was unable to be cancelled.")
        after = await prefect_client.read_flow_run(before.id)
        assert after.state.name == before.state.name
        assert after.state.type == before.state.type

    def test_wrong_id_exits_with_error(self) -> None:
        bad_id = str(uuid4())
        invoke_and_assert(['flow-run', 'cancel', bad_id], expected_code=1, expected_output_contains=f"Flow run '{bad_id}' not found!\n")

@pytest.fixture()
def flow_run_factory(prefect_client: PrefectClient) -> Callable[[int], Awaitable[FlowRun]]:

    async def create_flow_run(num_logs: int) -> FlowRun:
        flow_run = await prefect_client.create_flow_run(name='scheduled_flow_run', flow=hello_flow)
        logs = [LogCreate(name='prefect.flow_runs', level=20, message=f'Log {i} from flow_run {flow_run.id}.', timestamp=DateTime.now(), flow_run_id=flow_run.id) for i in range(num_logs)]
        await prefect_client.create_logs(logs)
        return flow_run
    return create_flow_run

class TestFlowRunLogs:
    LOGS_DEFAULT_PAGE_SIZE: int = 200

    async def test_when_num_logs_smaller_than_page_size_then_no_pagination(self, flow_run_factory: Callable[[int], Awaitable[FlowRun]]) -> None:
        flow_run = await flow_run_factory(num_logs=self.LOGS_DEFAULT_PAGE_SIZE - 1)
        await run_sync_in_worker_thread(invoke_and_assert, command=['flow-run', 'logs', str(flow_run.id)], expected_code=0, expected_output_contains=[f"Flow run '{flow_run.name}' - Log {i} from flow_run {flow_run.id}." for i in range(self.LOGS_DEFAULT_PAGE_SIZE - 1)])

    async def test_when_num_logs_greater_than_page_size_then_pagination(self, flow_run_factory: Callable[[int], Awaitable[FlowRun]]) -> None:
        flow_run = await flow_run_factory(num_logs=self.LOGS_DEFAULT_PAGE_SIZE + 1)
        await run_sync_in_worker_thread(invoke_and_assert, command=['flow-run', 'logs', str(flow_run.id)], expected_code=0, expected_output_contains=[f"Flow run '{flow_run.name}' - Log {i} from flow_run {flow_run.id}." for i in range(self.LOGS_DEFAULT_PAGE_SIZE + 1)])

    async def test_when_flow_run_not_found_then_exit_with_error(self, flow_run_factory: Callable[[int], Awaitable[FlowRun]]) -> None:
        bad_id = str(uuid4())
        await run_sync_in_worker_thread(invoke_and_assert, command=['flow-run', 'logs', bad_id], expected_code=1, expected_output_contains=f"Flow run '{bad_id}' not found!\n")

    async def test_when_num_logs_smaller_than_page_size_with_head_then_no_pagination(self, flow_run_factory: Callable[[int], Awaitable[FlowRun]]) -> None:
        flow_run = await flow_run_factory(num_logs=self.LOGS_DEFAULT_PAGE_SIZE + 1)
        await run_sync_in_worker_thread(invoke_and_assert, command=['flow-run', 'logs', str(flow_run.id), '--head', '--num-logs', '10'], expected_code=0, expected_output_contains=[f"Flow run '{flow_run.name}' - Log {i} from flow_run {flow_run.id}." for i in range(10)], expected_line_count=10)

    async def test_when_num_logs_greater_than_page_size_with_head_then_pagination(self, flow_run_factory: Callable[[int], Awaitable[FlowRun]]) -> None:
        flow_run = await flow_run_factory(num_logs=self.LOGS_DEFAULT_PAGE_SIZE + 1)
        await run_sync_in_worker_thread(invoke_and_assert, command=['flow-run', 'logs', str(flow_run.id), '--head', '--num-logs', self.LOGS_DEFAULT_PAGE_SIZE + 1], expected_code=0, expected_output_contains=[f"Flow run '{flow_run.name}' - Log {i} from flow_run {flow_run.id}." for i in range(self.LOGS_DEFAULT_PAGE_SIZE + 1)], expected_line_count=self.LOGS_DEFAULT_PAGE_SIZE + 1)

    async def test_when_num_logs_greater_than_page_size_with_head_outputs_correct_num_logs(self, flow_run_factory: Callable[[int], Awaitable[FlowRun]]) -> None:
        flow_run = await flow_run_factory(num_logs=self.LOGS_DEFAULT_PAGE_SIZE + 50)
        await run_sync_in_worker_thread(invoke_and_assert, command=['flow-run', 'logs', str(flow_run.id), '--head', '--num-logs', self.LOGS_DEFAULT_PAGE_SIZE + 50], expected_code=0, expected_output_contains=[f"Flow run '{flow_run.name}' - Log {i} from flow_run {flow_run.id}." for i in range(self.LOGS_DEFAULT_PAGE_SIZE + 50)], expected_line_count=self.LOGS_DEFAULT_PAGE_SIZE + 50)

    async def test_default_head_returns_default_num_logs(self, flow_run_factory: Callable[[int], Awaitable[FlowRun]]) -> None:
        flow_run = await flow_run_factory(num_logs=self.LOGS_DEFAULT_PAGE_SIZE + 1)
        await run_sync_in_worker_thread(invoke_and_assert, command=['flow-run', 'logs', str(flow_run.id), '--head'], expected_code=0, expected_output_contains=[f"Flow run '{flow_run.name}' - Log {i} from flow_run {flow_run.id}." for i in range(LOGS_WITH_LIMIT_FLAG_DEFAULT_NUM_LOGS)], expected_line_count=LOGS_WITH_LIMIT_FLAG_DEFAULT_NUM_LOGS)

    async def test_h_and_n_shortcuts_for_head_and_num_logs(self, flow_run_factory: Callable[[int], Awaitable[FlowRun]]) -> None:
        flow_run = await flow_run_factory(num_logs=self.LOGS_DEFAULT_PAGE_SIZE + 1)
        await run_sync_in_worker_thread(invoke_and_assert, command=['flow-run', 'logs', str(flow_run.id), '-h', '-n', '10'], expected_code=0, expected_output_contains=[f"Flow run '{flow_run.name}' - Log {i} from flow_run {flow_run.id}." for i in range(10)], expected_line_count=10)

    async def test_num_logs_passed_standalone_returns_num_logs(self, flow_run_factory: Callable[[int], Awaitable[FlowRun]]) -> None:
        flow_run = await flow_run_factory(num_logs=self.LOGS_DEFAULT_PAGE_SIZE + 1)
        await run_sync_in_worker_thread(invoke_and_assert, command=['flow-run', 'logs', str(flow_run.id), '--num-logs', '10'], expected_code=0, expected_output_contains=[f"Flow run '{flow_run.name}' - Log {i} from flow_run {flow_run.id}." for i in range(10)], expected_line_count=10)

    @pytest.mark.skip(reason='we need to disable colors for this test to pass')
    async def test_when_num_logs_is_smaller_than_one_then_exit_with_error(self, flow_run_factory: Callable[[int], Awaitable[FlowRun]]) -> None:
        flow_run = await flow_run_factory(num_logs=self.LOGS_DEFAULT_PAGE_SIZE + 1)
        await run_sync_in_worker_thread(invoke_and_assert, command=['flow-run', 'logs', str(flow_run.id), '--num-logs', '0'], expected_code=2, expected_output_contains="Invalid value for '--num-logs' / '-n':