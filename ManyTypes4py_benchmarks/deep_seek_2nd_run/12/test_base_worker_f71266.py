import uuid
from typing import Any, Dict, Optional, Type, List, Set, Union, cast
from unittest import mock
from unittest.mock import MagicMock, Mock
import httpx
import pendulum
import pytest
from packaging import version
from pydantic import Field
from starlette import status
import prefect
import prefect.client.schemas as schemas
from prefect.blocks.core import Block
from prefect.client.base import ServerType
from prefect.client.orchestration import PrefectClient, get_client
from prefect.client.schemas import FlowRun
from prefect.client.schemas.objects import Integration, StateType, WorkerMetadata
from prefect.exceptions import CrashedRun, ObjectNotFound
from prefect.flows import flow
from prefect.server import models
from prefect.server.schemas.actions import WorkPoolUpdate as ServerWorkPoolUpdate
from prefect.server.schemas.core import Deployment, Flow, WorkPool
from prefect.server.schemas.responses import DeploymentResponse
from prefect.settings import PREFECT_API_URL, PREFECT_TEST_MODE, PREFECT_WORKER_PREFETCH_SECONDS, get_current_settings, temporary_settings
from prefect.states import Completed, Failed, Pending, Running, Scheduled
from prefect.testing.utilities import AsyncMock
from prefect.utilities.pydantic import parse_obj_as
from prefect.workers.base import BaseJobConfiguration, BaseVariables, BaseWorker
from prefect.logging import Logger
from sqlalchemy.ext.asyncio import AsyncSession
from prefect.server.database.orm_models import ORMWorkPool

class WorkerTestImpl(BaseWorker):
    type: str = 'test'
    job_configuration: Type[BaseJobConfiguration] = BaseJobConfiguration

    async def run(self, flow_run: FlowRun, configuration: BaseJobConfiguration) -> None:
        pass

@pytest.fixture(autouse=True)
async def ensure_default_agent_pool_exists(session: AsyncSession) -> None:
    default_work_pool: Optional[ORMWorkPool] = await models.workers.read_work_pool_by_name(session=session, work_pool_name=models.workers.DEFAULT_AGENT_WORK_POOL_NAME)
    if default_work_pool is None:
        await models.workers.create_work_pool(session=session, work_pool=WorkPool(name=models.workers.DEFAULT_AGENT_WORK_POOL_NAME, type='prefect-agent'))
        await session.commit()

@pytest.fixture
async def variables(prefect_client: PrefectClient) -> None:
    await prefect_client._client.post('/variables/', json={'name': 'test_variable_1', 'value': 'test_value_1'})
    await prefect_client._client.post('/variables/', json={'name': 'test_variable_2', 'value': 'test_value_2'})

@pytest.fixture
def no_api_url() -> Any:
    with temporary_settings(updates={PREFECT_TEST_MODE: False, PREFECT_API_URL: None}):
        yield

async def test_worker_requires_api_url_when_not_in_test_mode(no_api_url: Any) -> None:
    with pytest.raises(ValueError, match='PREFECT_API_URL'):
        async with WorkerTestImpl(name='test', work_pool_name='test-work-pool'):
            pass

async def test_worker_creates_work_pool_by_default_during_sync(prefect_client: PrefectClient) -> None:
    with pytest.raises(ObjectNotFound):
        await prefect_client.read_work_pool('test-work-pool')
    async with WorkerTestImpl(name='test', work_pool_name='test-work-pool') as worker:
        await worker.sync_with_backend()
        worker_status: Dict[str, Any] = worker.get_status()
        assert worker_status['work_pool']['name'] == 'test-work-pool'
        work_pool: WorkPool = await prefect_client.read_work_pool('test-work-pool')
        assert str(work_pool.id) == worker_status['work_pool']['id']

async def test_worker_does_not_creates_work_pool_when_create_pool_is_false(prefect_client: PrefectClient) -> None:
    with pytest.raises(ObjectNotFound):
        await prefect_client.read_work_pool('test-work-pool')
    async with WorkerTestImpl(name='test', work_pool_name='test-work-pool', create_pool_if_not_found=False) as worker:
        await worker.sync_with_backend()
        worker_status: Dict[str, Any] = worker.get_status()
        assert worker_status['work_pool'] is None
    with pytest.raises(ObjectNotFound):
        await prefect_client.read_work_pool('test-work-pool')

@pytest.mark.parametrize('setting,attr', [(PREFECT_WORKER_PREFETCH_SECONDS, 'prefetch_seconds')])
async def test_worker_respects_settings(setting: Any, attr: str) -> None:
    assert WorkerTestImpl(name='test', work_pool_name='test-work-pool').get_status()['settings'][attr] == setting.value()

async def test_worker_sends_heartbeat_messages(prefect_client: PrefectClient) -> None:
    async with WorkerTestImpl(name='test', work_pool_name='test-work-pool') as worker:
        await worker.sync_with_backend()
        workers: List[schemas.objects.Worker] = await prefect_client.read_workers_for_work_pool(work_pool_name='test-work-pool')
        assert len(workers) == 1
        first_heartbeat: Optional[pendulum.DateTime] = workers[0].last_heartbeat_time
        assert first_heartbeat is not None
        await worker.sync_with_backend()
        workers = await prefect_client.read_workers_for_work_pool(work_pool_name='test-work-pool')
        second_heartbeat: Optional[pendulum.DateTime] = workers[0].last_heartbeat_time
        assert second_heartbeat > first_heartbeat

async def test_worker_sends_heartbeat_gets_id(respx_mock: Any) -> None:
    work_pool_name: str = 'test-work-pool'
    test_worker_id: uuid.UUID = uuid.UUID('028EC481-5899-49D7-B8C5-37A2726E9840')
    async with WorkerTestImpl(name='test', work_pool_name=work_pool_name) as worker:
        setattr(worker, '_should_get_worker_id', lambda: True)
        respx_mock.get(f'api/work_pools/{work_pool_name}').pass_through()
        respx_mock.get('api/csrf-token?').pass_through()
        respx_mock.post('api/work_pools/').pass_through()
        respx_mock.patch(f'api/work_pools/{work_pool_name}').pass_through()
        respx_mock.post(f'api/work_pools/{work_pool_name}/workers/heartbeat').mock(return_value=httpx.Response(status.HTTP_200_OK, text=str(test_worker_id)))
        await worker.sync_with_backend()
        assert worker.backend_id == test_worker_id

async def test_worker_sends_heartbeat_only_gets_id_once() -> None:
    async with WorkerTestImpl(name='test', work_pool_name='test-work-pool') as worker:
        worker._client.server_type = ServerType.CLOUD
        mock: AsyncMock = AsyncMock(return_value='test')
        setattr(worker._client, 'send_worker_heartbeat', mock)
        await worker.sync_with_backend()
        await worker.sync_with_backend()
        second_call: mock.call = mock.await_args_list[1]
        assert worker.backend_id == 'test'
        assert not second_call.kwargs['get_worker_id']

async def test_worker_with_work_pool(prefect_client: PrefectClient, worker_deployment_wq1: DeploymentResponse, work_pool: WorkPool) -> None:

    @flow
    def test_flow() -> None:
        pass

    def create_run_with_deployment(state: schemas.states.State) -> Any:
        return prefect_client.create_flow_run_from_deployment(worker_deployment_wq1.id, state=state)
    flow_runs: List[FlowRun] = [await create_run_with_deployment(Pending()), await create_run_with_deployment(Scheduled(scheduled_time=pendulum.now('utc').subtract(days=1))), await create_run_with_deployment(Scheduled(scheduled_time=pendulum.now('utc').add(seconds=5))), await create_run_with_deployment(Scheduled(scheduled_time=pendulum.now('utc').add(seconds=5))), await create_run_with_deployment(Scheduled(scheduled_time=pendulum.now('utc').add(seconds=20))), await create_run_with_deployment(Running()), await create_run_with_deployment(Completed()), await prefect_client.create_flow_run(test_flow, state=Scheduled())]
    flow_run_ids: List[uuid.UUID] = [run.id for run in flow_runs]
    async with WorkerTestImpl(work_pool_name=work_pool.name) as worker:
        submitted_flow_runs: List[FlowRun] = await worker.get_and_submit_flow_runs()
    assert {flow_run.id for flow_run in submitted_flow_runs} == set(flow_run_ids[1:4])

async def test_worker_with_work_pool_and_work_queue(prefect_client: PrefectClient, worker_deployment_wq1: DeploymentResponse, worker_deployment_wq_2: DeploymentResponse, work_queue_1: schemas.objects.WorkQueue, work_pool: WorkPool) -> None:

    @flow
    def test_flow() -> None:
        pass

    def create_run_with_deployment_1(state: schemas.states.State) -> Any:
        return prefect_client.create_flow_run_from_deployment(worker_deployment_wq1.id, state=state)

    def create_run_with_deployment_2(state: schemas.states.State) -> Any:
        return prefect_client.create_flow_run_from_deployment(worker_deployment_wq_2.id, state=state)
    flow_runs: List[FlowRun] = [await create_run_with_deployment_1(Pending()), await create_run_with_deployment_1(Scheduled(scheduled_time=pendulum.now('utc').subtract(days=1))), await create_run_with_deployment_1(Scheduled(scheduled_time=pendulum.now('utc').add(seconds=5))), await create_run_with_deployment_2(Scheduled(scheduled_time=pendulum.now('utc').add(seconds=5))), await create_run_with_deployment_2(Scheduled(scheduled_time=pendulum.now('utc').add(seconds=20))), await create_run_with_deployment_1(Running()), await create_run_with_deployment_1(Completed()), await prefect_client.create_flow_run(test_flow, state=Scheduled())]
    flow_run_ids: List[uuid.UUID] = [run.id for run in flow_runs]
    async with WorkerTestImpl(work_pool_name=work_pool.name, work_queues=[work_queue_1.name]) as worker:
        submitted_flow_runs: List[FlowRun] = await worker.get_and_submit_flow_runs()
    assert {flow_run.id for flow_run in submitted_flow_runs} == set(flow_run_ids[1:3])

async def test_workers_do_not_submit_flow_runs_awaiting_retry(prefect_client: PrefectClient, work_queue_1: schemas.objects.WorkQueue, work_pool: WorkPool) -> None:
    """
    Regression test for https://github.com/PrefectHQ/prefect/issues/15458
    """

    @flow(retries=2)
    def test_flow() -> None:
        pass
    flow_id: uuid.UUID = await prefect_client.create_flow(flow=test_flow)
    deployment_id: uuid.UUID = await prefect_client.create_deployment(flow_id=flow_id, name='test-deployment', work_queue_name=work_queue_1.name, work_pool_name=work_pool.name)
    flow_run: FlowRun = await prefect_client.create_flow_run_from_deployment(deployment_id, state=Running())
    flow_run.empirical_policy.retries = 2
    await prefect_client.update_flow_run(flow_run_id=flow_run.id, flow_version=test_flow.version, empirical_policy=flow_run.empirical_policy)
    response: schemas.responses.SetStateResponse = await prefect_client.set_flow_run_state(flow_run.id, state=Failed())
    assert response.state.name == 'AwaitingRetry'
    assert response.state.type == StateType.SCHEDULED
    flow_run = await prefect_client.read_flow_run(flow_run.id)
    assert flow_run.state.state_details.scheduled_time < pendulum.now('utc')
    async with WorkerTestImpl(work_pool_name=work_pool.name) as worker:
        submitted_flow_runs: List[FlowRun] = await worker.get_and_submit_flow_runs()
    assert submitted_flow_runs == []

async def test_priority_trumps_lateness(prefect_client: PrefectClient, worker_deployment_wq1: DeploymentResponse, worker_deployment_wq_2: DeploymentResponse, work_queue_1: schemas.objects.WorkQueue, work_pool: WorkPool) -> None:

    @flow
    def test_flow() -> None:
        pass

    def create_run_with_deployment_1(state: schemas.states.State) -> Any:
        return prefect_client.create_flow_run_from_deployment(worker_deployment_wq1.id, state=state)

    def create_run_with_deployment_2(state: schemas.states.State) -> Any:
        return prefect_client.create_flow_run_from_deployment(worker_deployment_wq_2.id, state=state)
    flow_runs: List[FlowRun] = [await create_run_with_deployment_2(Scheduled(scheduled_time=pendulum.now('utc').subtract(days=1))), await create_run_with_deployment_1(Scheduled(scheduled_time=pendulum.now('utc').add(seconds=5)))]
    flow_run_ids: List[uuid.UUID] = [run.id for run in flow_runs]
    async with WorkerTestImpl(work_pool_name=work_pool.name, limit=1) as worker:
        worker._submit_run = AsyncMock()
        submitted_flow_runs: List[FlowRun] = await worker.get_and_submit_flow_runs()
    assert {flow_run.id for flow_run in submitted_flow_runs} == set(flow_run_ids[1:2])

async def test_worker_releases_limit_slot_when_aborting_a_change_to_pending(prefect_client: PrefectClient, worker_deployment_wq1: DeploymentResponse, work_pool: WorkPool) -> None:
    """Regression test for https://github.com/PrefectHQ/prefect/issues/15952"""

    def create_run_with_deployment(state: schemas.states.State) -> Any:
        return prefect_client.create_flow_run_from_deployment(worker_deployment_wq1.id, state=state)
    flow_run: FlowRun = await create_run_with_deployment(Scheduled(scheduled_time=pendulum.now('utc').subtract(days=1)))
    run_mock: AsyncMock = AsyncMock()
    release_mock: Mock = Mock()
    async with WorkerTestImpl(work_pool_name=work_pool.name, limit=1) as worker:
        worker.run = run_mock
        worker._propose_pending_state = AsyncMock(return_value=False)
        worker._release_limit_slot = release_mock
        await worker.get_and_submit_flow_runs()
    run_mock.assert_not_called()
    release_mock.assert_called_once_with(flow_run.id)

async def test_worker_with_work_pool_and_limit(prefect_client: PrefectClient, worker_deployment_wq1: DeploymentResponse, work_pool: WorkPool) -> None:

    @flow
    def test_flow() -> None:
        pass

    def create_run_with_deployment(state: schemas.states.State) -> Any:
        return prefect_client.create_flow_run_from_deployment(worker_deployment_wq1.id, state=state)
    flow_runs: List[FlowRun] = [await create_run_with_deployment(Pending()), await create_run_with_deployment(Scheduled(scheduled_time=pendulum.now('utc').subtract(days=1))), await create_run_with_deployment(Scheduled(scheduled_time=pendulum.now('utc').add(seconds=5))), await create_run_with_deployment(Scheduled(scheduled_time=pendulum.now('utc').add(seconds=5))), await create_run_with_deployment(Scheduled(scheduled_time=pendulum.now('utc').add(seconds=20))), await create_run_with_deployment(Running()), await create_run_with_deployment(Completed()), await prefect_client.create_flow_run(test_flow, state=Scheduled())]
    flow_run_ids: List[uuid.UUID] = [run.id for run in flow_runs]
    async with WorkerTestImpl(work_pool_name=work_pool.name, limit=2) as worker:
        worker._submit_run = AsyncMock()
        submitted_flow_runs: List[FlowRun] = await worker.get_and_submit_flow_runs()
        assert {flow_run.id for flow_run in submitted_flow_runs} == set(flow_run_ids[1:3])
        submitted_flow_runs = await worker.get_and_submit_flow_runs()
        assert {flow_run.id for flow_run in submitted_flow_runs} == set(flow_run_ids[1:3])
        worker._limiter.release_on_behalf_of(flow_run_ids[1])
        submitted_flow_runs = await worker.get_and_submit_flow_runs()
        assert {flow_run.id for flow_run in submitted_flow_runs} == set(flow_run_ids[1:4])

async def test_worker_calls_run_with_expected_arguments(prefect_client: PrefectClient, worker_deployment_wq1: DeploymentResponse, work_pool: WorkPool, monkeypatch: pytest.MonkeyPatch) -> None:
    run_mock: AsyncMock = AsyncMock()

    @flow
    def test_flow() -> None:
        pass

    def create_run_with_deployment(state: schemas.states.State) ->