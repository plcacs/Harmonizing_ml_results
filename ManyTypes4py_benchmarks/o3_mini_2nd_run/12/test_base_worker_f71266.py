#!/usr/bin/env python3
import uuid
from typing import Any, AsyncGenerator, Dict, Generator, Optional, Type
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


class WorkerTestImpl(BaseWorker):
    type: str = 'test'
    job_configuration: Type[BaseJobConfiguration] = BaseJobConfiguration

    async def run(self) -> None:
        pass


@pytest.fixture(autouse=True)
async def ensure_default_agent_pool_exists(session: Any) -> None:
    default_work_pool = await models.workers.read_work_pool_by_name(
        session=session, work_pool_name=models.workers.DEFAULT_AGENT_WORK_POOL_NAME
    )
    if default_work_pool is None:
        await models.workers.create_work_pool(
            session=session,
            work_pool=WorkPool(name=models.workers.DEFAULT_AGENT_WORK_POOL_NAME, type='prefect-agent')
        )
        await session.commit()


@pytest.fixture
async def variables(prefect_client: PrefectClient) -> None:
    await prefect_client._client.post('/variables/', json={'name': 'test_variable_1', 'value': 'test_value_1'})
    await prefect_client._client.post('/variables/', json={'name': 'test_variable_2', 'value': 'test_value_2'})


@pytest.fixture
def no_api_url() -> Generator[None, None, None]:
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
        work_pool = await prefect_client.read_work_pool('test-work-pool')
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
        workers = await prefect_client.read_workers_for_work_pool(work_pool_name='test-work-pool')
        assert len(workers) == 1
        first_heartbeat = workers[0].last_heartbeat_time
        assert first_heartbeat is not None
        await worker.sync_with_backend()
        workers = await prefect_client.read_workers_for_work_pool(work_pool_name='test-work-pool')
        second_heartbeat = workers[0].last_heartbeat_time
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
        respx_mock.post(f'api/work_pools/{work_pool_name}/workers/heartbeat').mock(
            return_value=httpx.Response(status.HTTP_200_OK, text=str(test_worker_id))
        )
        await worker.sync_with_backend()
        assert worker.backend_id == test_worker_id


async def test_worker_sends_heartbeat_only_gets_id_once() -> None:
    async with WorkerTestImpl(name='test', work_pool_name='test-work-pool') as worker:
        worker._client.server_type = ServerType.CLOUD
        mock_send = AsyncMock(return_value='test')
        setattr(worker._client, 'send_worker_heartbeat', mock_send)
        await worker.sync_with_backend()
        await worker.sync_with_backend()
        second_call = mock_send.await_args_list[1]
        assert worker.backend_id == 'test'
        assert not second_call.kwargs['get_worker_id']


async def test_worker_with_work_pool(prefect_client: PrefectClient, worker_deployment_wq1: Any, work_pool: WorkPool) -> None:
    @flow
    def test_flow() -> None:
        pass

    def create_run_with_deployment(state: Any) -> Any:
        return prefect_client.create_flow_run_from_deployment(worker_deployment_wq1.id, state=state)

    flow_runs = [
        await create_run_with_deployment(Pending()),
        await create_run_with_deployment(Scheduled(scheduled_time=pendulum.now('utc').subtract(days=1))),
        await create_run_with_deployment(Scheduled(scheduled_time=pendulum.now('utc').add(seconds=5))),
        await create_run_with_deployment(Scheduled(scheduled_time=pendulum.now('utc').add(seconds=5))),
        await create_run_with_deployment(Scheduled(scheduled_time=pendulum.now('utc').add(seconds=20))),
        await create_run_with_deployment(Running()),
        await create_run_with_deployment(Completed()),
        await prefect_client.create_flow_run(test_flow, state=Scheduled())
    ]
    flow_run_ids = [run.id for run in flow_runs]
    async with WorkerTestImpl(work_pool_name=work_pool.name) as worker:
        submitted_flow_runs = await worker.get_and_submit_flow_runs()
    assert {flow_run.id for flow_run in submitted_flow_runs} == set(flow_run_ids[1:4])


async def test_worker_with_work_pool_and_work_queue(prefect_client: PrefectClient, worker_deployment_wq1: Any, worker_deployment_wq_2: Any, work_queue_1: Any, work_pool: WorkPool) -> None:
    @flow
    def test_flow() -> None:
        pass

    def create_run_with_deployment_1(state: Any) -> Any:
        return prefect_client.create_flow_run_from_deployment(worker_deployment_wq1.id, state=state)

    def create_run_with_deployment_2(state: Any) -> Any:
        return prefect_client.create_flow_run_from_deployment(worker_deployment_wq_2.id, state=state)

    flow_runs = [
        await create_run_with_deployment_1(Pending()),
        await create_run_with_deployment_1(Scheduled(scheduled_time=pendulum.now('utc').subtract(days=1))),
        await create_run_with_deployment_1(Scheduled(scheduled_time=pendulum.now('utc').add(seconds=5))),
        await create_run_with_deployment_2(Scheduled(scheduled_time=pendulum.now('utc').add(seconds=5))),
        await create_run_with_deployment_2(Scheduled(scheduled_time=pendulum.now('utc').add(seconds=20))),
        await create_run_with_deployment_1(Running()),
        await create_run_with_deployment_1(Completed()),
        await prefect_client.create_flow_run(test_flow, state=Scheduled())
    ]
    flow_run_ids = [run.id for run in flow_runs]
    async with WorkerTestImpl(work_pool_name=work_pool.name, work_queues=[work_queue_1.name]) as worker:
        submitted_flow_runs = await worker.get_and_submit_flow_runs()
    assert {flow_run.id for flow_run in submitted_flow_runs} == set(flow_run_ids[1:3])


async def test_workers_do_not_submit_flow_runs_awaiting_retry(prefect_client: PrefectClient, work_queue_1: Any, work_pool: WorkPool) -> None:
    """
    Regression test for https://github.com/PrefectHQ/prefect/issues/15458

    Ensure that flows in `AwaitingRetry` state are not submitted by workers. Previously,
    with a retry delay long enough, workers would pick up flow runs in `AwaitingRetry`
    state and submit them, even though the process they were initiated from is responsible
    for retrying them.

    The flows would be picked up by the worker because `AwaitingRetry` is a `SCHEDULED`
    state type.
    """
    @flow(retries=2)
    def test_flow() -> None:
        pass
    flow_id = await prefect_client.create_flow(flow=test_flow)
    deployment_id = await prefect_client.create_deployment(
        flow_id=flow_id, name='test-deployment', work_queue_name=work_queue_1.name, work_pool_name=work_pool.name
    )
    flow_run = await prefect_client.create_flow_run_from_deployment(deployment_id, state=Running())
    flow_run.empirical_policy.retries = 2
    await prefect_client.update_flow_run(
        flow_run_id=flow_run.id, flow_version=test_flow.version, empirical_policy=flow_run.empirical_policy
    )
    response = await prefect_client.set_flow_run_state(flow_run.id, state=Failed())
    assert response.state.name == 'AwaitingRetry'
    assert response.state.type == StateType.SCHEDULED
    flow_run = await prefect_client.read_flow_run(flow_run.id)
    assert flow_run.state.state_details.scheduled_time < pendulum.now('utc')
    async with WorkerTestImpl(work_pool_name=work_pool.name) as worker:
        submitted_flow_runs = await worker.get_and_submit_flow_runs()
    assert submitted_flow_runs == []


async def test_priority_trumps_lateness(prefect_client: PrefectClient, worker_deployment_wq1: Any, worker_deployment_wq_2: Any, work_queue_1: Any, work_pool: WorkPool) -> None:
    @flow
    def test_flow() -> None:
        pass

    def create_run_with_deployment_1(state: Any) -> Any:
        return prefect_client.create_flow_run_from_deployment(worker_deployment_wq1.id, state=state)

    def create_run_with_deployment_2(state: Any) -> Any:
        return prefect_client.create_flow_run_from_deployment(worker_deployment_wq_2.id, state=state)

    flow_runs = [
        await create_run_with_deployment_2(Scheduled(scheduled_time=pendulum.now('utc').subtract(days=1))),
        await create_run_with_deployment_1(Scheduled(scheduled_time=pendulum.now('utc').add(seconds=5)))
    ]
    flow_run_ids = [run.id for run in flow_runs]
    async with WorkerTestImpl(work_pool_name=work_pool.name, limit=1) as worker:
        worker._submit_run = AsyncMock()
        submitted_flow_runs = await worker.get_and_submit_flow_runs()
    assert {flow_run.id for flow_run in submitted_flow_runs} == set(flow_run_ids[1:2])


async def test_worker_releases_limit_slot_when_aborting_a_change_to_pending(prefect_client: PrefectClient, worker_deployment_wq1: Any, work_pool: WorkPool) -> None:
    """Regression test for https://github.com/PrefectHQ/prefect/issues/15952"""
    def create_run_with_deployment(state: Any) -> Any:
        return prefect_client.create_flow_run_from_deployment(worker_deployment_wq1.id, state=state)
    flow_run = await create_run_with_deployment(Scheduled(scheduled_time=pendulum.now('utc').subtract(days=1)))
    run_mock: AsyncMock = AsyncMock()
    release_mock: Any = Mock()
    async with WorkerTestImpl(work_pool_name=work_pool.name, limit=1) as worker:
        worker.run = run_mock
        worker._propose_pending_state = AsyncMock(return_value=False)
        worker._release_limit_slot = release_mock
        await worker.get_and_submit_flow_runs()
    run_mock.assert_not_called()
    release_mock.assert_called_once_with(flow_run.id)


async def test_worker_with_work_pool_and_limit(prefect_client: PrefectClient, worker_deployment_wq1: Any, work_pool: WorkPool) -> None:
    @flow
    def test_flow() -> None:
        pass

    def create_run_with_deployment(state: Any) -> Any:
        return prefect_client.create_flow_run_from_deployment(worker_deployment_wq1.id, state=state)

    flow_runs = [
        await create_run_with_deployment(Pending()),
        await create_run_with_deployment(Scheduled(scheduled_time=pendulum.now('utc').subtract(days=1))),
        await create_run_with_deployment(Scheduled(scheduled_time=pendulum.now('utc').add(seconds=5))),
        await create_run_with_deployment(Scheduled(scheduled_time=pendulum.now('utc').add(seconds=5))),
        await create_run_with_deployment(Scheduled(scheduled_time=pendulum.now('utc').add(seconds=20))),
        await create_run_with_deployment(Running()),
        await create_run_with_deployment(Completed()),
        await prefect_client.create_flow_run(test_flow, state=Scheduled())
    ]
    flow_run_ids = [run.id for run in flow_runs]
    async with WorkerTestImpl(work_pool_name=work_pool.name, limit=2) as worker:
        worker._submit_run = AsyncMock()
        submitted_flow_runs = await worker.get_and_submit_flow_runs()
        assert {flow_run.id for flow_run in submitted_flow_runs} == set(flow_run_ids[1:3])
        submitted_flow_runs = await worker.get_and_submit_flow_runs()
        assert {flow_run.id for flow_run in submitted_flow_runs} == set(flow_run_ids[1:3])
        worker._limiter.release_on_behalf_of(flow_run_ids[1])
        submitted_flow_runs = await worker.get_and_submit_flow_runs()
        assert {flow_run.id for flow_run in submitted_flow_runs} == set(flow_run_ids[1:4])


async def test_worker_calls_run_with_expected_arguments(prefect_client: PrefectClient, worker_deployment_wq1: Any, work_pool: WorkPool, monkeypatch: Any) -> None:
    run_mock: AsyncMock = AsyncMock()

    @flow
    def test_flow() -> None:
        pass

    def create_run_with_deployment(state: Any) -> Any:
        return prefect_client.create_flow_run_from_deployment(worker_deployment_wq1.id, state=state)

    flow_runs = [
        await create_run_with_deployment(Pending()),
        await create_run_with_deployment(Scheduled(scheduled_time=pendulum.now('utc').subtract(days=1))),
        await create_run_with_deployment(Scheduled(scheduled_time=pendulum.now('utc').add(seconds=5))),
        await create_run_with_deployment(Scheduled(scheduled_time=pendulum.now('utc').add(seconds=5))),
        await create_run_with_deployment(Scheduled(scheduled_time=pendulum.now('utc').add(seconds=20))),
        await create_run_with_deployment(Running()),
        await create_run_with_deployment(Completed()),
        await prefect_client.create_flow_run(test_flow, state=Scheduled())
    ]
    async with WorkerTestImpl(work_pool_name=work_pool.name) as worker:
        worker._work_pool = work_pool
        worker.run = run_mock
        await worker.get_and_submit_flow_runs()
    assert run_mock.call_count == 3
    assert {call.kwargs['flow_run'].id for call in run_mock.call_args_list} == {fr.id for fr in flow_runs[1:4]}


async def test_worker_warns_when_running_a_flow_run_with_a_storage_block(prefect_client: PrefectClient, deployment: Deployment, work_pool: WorkPool, caplog: Any) -> None:
    @flow
    def test_flow() -> None:
        pass

    def create_run_with_deployment(state: Any) -> Any:
        return prefect_client.create_flow_run_from_deployment(deployment.id, state=state)
    flow_run = await create_run_with_deployment(Scheduled(scheduled_time=pendulum.now('utc').add(seconds=5)))
    async with WorkerTestImpl(work_pool_name=work_pool.name) as worker:
        worker._work_pool = work_pool
        await worker.get_and_submit_flow_runs()
    assert f'Flow run {flow_run.id!r} was created from deployment {deployment.name!r} which is configured with a storage block. Please use an' + ' agent to execute this flow run.' in caplog.text
    flow_run = await prefect_client.read_flow_run(flow_run.id)
    assert flow_run.state_name == 'Scheduled'


async def test_worker_creates_only_one_client_context(prefect_client: PrefectClient, worker_deployment_wq1: Any, work_pool: WorkPool, monkeypatch: Any, caplog: Any) -> None:
    tracking_mock = MagicMock()
    orig_get_client = get_client

    def get_client_spy(*args: Any, **kwargs: Any) -> Any:
        tracking_mock(*args, **kwargs)
        return orig_get_client(*args, **kwargs)
    monkeypatch.setattr('prefect.workers.base.get_client', get_client_spy)
    run_mock: AsyncMock = AsyncMock()

    @flow
    def test_flow() -> None:
        pass

    def create_run_with_deployment(state: Any) -> Any:
        return prefect_client.create_flow_run_from_deployment(worker_deployment_wq1.id, state=state)
    await create_run_with_deployment(Scheduled(scheduled_time=pendulum.now('utc').subtract(days=1)))
    await create_run_with_deployment(Scheduled(scheduled_time=pendulum.now('utc').add(seconds=5)))
    await create_run_with_deployment(Scheduled(scheduled_time=pendulum.now('utc').add(seconds=5)))
    async with WorkerTestImpl(work_pool_name=work_pool.name) as worker:
        worker._work_pool = work_pool
        worker.run = run_mock
        await worker.get_and_submit_flow_runs()
    assert tracking_mock.call_count == 1


async def test_base_worker_gets_job_configuration_when_syncing_with_backend_with_just_job_config(session: Any, client: Any) -> None:
    """We don't really care how this happens as long as the worker winds up with a worker pool
    with a correct base_job_template when creating a new work pool"""
    class WorkerJobConfig(BaseJobConfiguration):
        other: Optional[Any] = Field(default=None, json_schema_extra={'template': '{{ other }}'})
    WorkerTestImpl.job_configuration = WorkerJobConfig
    expected_job_template: Dict[str, Any] = {
        'job_configuration': {
            'command': '{{ command }}',
            'env': '{{ env }}',
            'labels': '{{ labels }}',
            'name': '{{ name }}',
            'other': '{{ other }}'
        },
        'variables': {
            'properties': {
                'command': {
                    'anyOf': [{'type': 'string'}, {'type': 'null'}],
                    'default': None,
                    'title': 'Command',
                    'description': 'The command to use when starting a flow run. In most cases, this should be left blank and the command will be automatically generated by the worker.'
                },
                'env': {
                    'title': 'Environment Variables',
                    'type': 'object',
                    'additionalProperties': {'anyOf': [{'type': 'string'}, {'type': 'null'}]},
                    'description': 'Environment variables to set when starting a flow run.'
                },
                'labels': {
                    'title': 'Labels',
                    'type': 'object',
                    'additionalProperties': {'type': 'string'},
                    'description': 'Labels applied to infrastructure created by the worker using this job configuration.'
                },
                'name': {
                    'anyOf': [{'type': 'string'}, {'type': 'null'}],
                    'default': None,
                    'title': 'Name',
                    'description': 'Name given to infrastructure created by the worker using this job configuration.'
                }
            },
            'type': 'object'
        }
    }
    pool_name: str = 'test-pool'
    response = await client.post('/work_pools/', json=dict(name=pool_name, type='test-type'))
    result = parse_obj_as(schemas.objects.WorkPool, response.json())
    model = await models.workers.read_work_pool(session=session, work_pool_id=result.id)
    assert model.name == pool_name
    worker = WorkerTestImpl(name='test', work_pool_name=pool_name)
    async with get_client() as client_context:
        worker._client = client_context
        await worker.sync_with_backend()
    assert worker._work_pool.base_job_template == expected_job_template


async def test_base_worker_gets_job_configuration_when_syncing_with_backend_with_job_config_and_variables(session: Any, client: Any) -> None:
    """We don't really care how this happens as long as the worker winds up with a worker pool
    with a correct base_job_template when creating a new work pool"""
    class WorkerJobConfig(BaseJobConfiguration):
        other: Optional[Any] = Field(default=None, json_schema_extra={'template': '{{ other }}'})

    class WorkerVariables(BaseVariables):
        other: str = Field(default='woof')
    WorkerTestImpl.job_configuration = WorkerJobConfig
    WorkerTestImpl.job_configuration_variables = WorkerVariables
    pool_name: str = 'test-pool'
    response = await client.post('/work_pools/', json=dict(name=pool_name, type='test-type'))
    result = parse_obj_as(schemas.objects.WorkPool, response.json())
    model = await models.workers.read_work_pool(session=session, work_pool_id=result.id)
    assert model.name == pool_name
    worker = WorkerTestImpl(name='test', work_pool_name=pool_name)
    async with get_client() as client_context:
        worker._client = client_context
        await worker.sync_with_backend()
    assert worker._work_pool.base_job_template == WorkerTestImpl.get_default_base_job_template()


@pytest.mark.parametrize('template,overrides,expected', [
    (
        {
            'job_configuration': {
                'command': '{{ command }}',
                'env': '{{ env }}',
                'labels': '{{ labels }}',
                'name': '{{ name }}'
            },
            'variables': {
                'properties': {
                    'command': {'type': 'string', 'title': 'Command', 'default': 'echo hello'},
                    'env': {
                        'title': 'Environment Variables',
                        'type': 'object',
                        'additionalProperties': {'type': 'string'},
                        'description': 'Environment variables to set when starting a flow run.'
                    }
                },
                'type': 'object'
            }
        },
        {},
        {'command': 'echo hello', 'env': {}, 'labels': {}, 'name': None}
    )
])
async def test_base_job_configuration_from_template_and_overrides(template: Dict[str, Any], overrides: Dict[str, Any], expected: Dict[str, Any]) -> None:
    config = await BaseJobConfiguration.from_template_and_values(base_job_template=template, values=overrides)
    assert config.model_dump() == expected


@pytest.mark.parametrize('template,overrides,expected', [
    (
        {
            'job_configuration': {'var1': '{{ var1 }}', 'var2': '{{ var2 }}'},
            'variables': {
                'properties': {
                    'var1': {'type': 'string', 'title': 'Var1', 'default': 'hello'},
                    'var2': {'type': 'integer', 'title': 'Var2', 'default': 42}
                },
                'required': []
            }
        },
        {},
        {'command': None, 'env': {}, 'labels': {}, 'name': None, 'var1': 'hello', 'var2': 42}
    ),
    (
        {
            'job_configuration': {'var1': '{{ var1 }}', 'var2': '{{ var2 }}'},
            'variables': {
                'properties': {
                    'var1': {'type': 'string', 'title': 'Var1', 'default': 'hello'},
                    'var2': {'type': 'integer', 'title': 'Var2', 'default': 42},
                    'var3': {'type': 'integer', 'title': 'Var3', 'default': 21}
                },
                'required': []
            }
        },
        {},
        {'command': None, 'env': {}, 'labels': {}, 'name': None, 'var1': 'hello', 'var2': 42}
    ),
    (
        {
            'job_configuration': {'var1': '{{ var1 }}', 'var2': '{{ var2 }}'},
            'variables': {
                'properties': {
                    'var1': {'type': 'string', 'title': 'Var1', 'default': 'hello'},
                    'var2': {'type': 'integer', 'title': 'Var2', 'default': 42},
                    'command': {'type': 'string', 'title': 'Command', 'default': 'echo hello'}
                },
                'required': []
            }
        },
        {},
        {'command': None, 'env': {}, 'labels': {}, 'name': None, 'var1': 'hello', 'var2': 42}
    ),
    (
        {
            'job_configuration': {'var1': '{{ var1 }}', 'var2': '{{ var2 }}'},
            'variables': {
                'properties': {
                    'var1': {'type': 'string', 'title': 'Var1', 'default': 'hello'},
                    'var2': {'type': 'integer', 'title': 'Var2', 'default': 42}
                }
            },
            'required': []
        },
        {'var1': 'woof!'},
        {'command': None, 'env': {}, 'labels': {}, 'name': None, 'var1': 'woof!', 'var2': 42}
    ),
    (
        {
            'job_configuration': {'var1': '{{ var1 }}', 'var2': '{{ var2 }}'},
            'variables': {
                'properties': {
                    'var1': {'type': 'string', 'title': 'Var1'},
                    'var2': {'type': 'integer', 'title': 'Var2', 'default': 42}
                },
            },
            'required': ['var1']
        },
        {'var1': 'woof!'},
        {'command': None, 'env': {}, 'labels': {}, 'name': None, 'var1': 'woof!', 'var2': 42}
    )
])
async def test_job_configuration_from_template_and_overrides(template: Dict[str, Any], overrides: Dict[str, Any], expected: Dict[str, Any]) -> None:
    class ArbitraryJobConfiguration(BaseJobConfiguration):
        var1: Optional[str] = Field(json_schema_extra={'template': '{{ var1 }}'})
        var2: Optional[int] = Field(json_schema_extra={'template': '{{ var2 }}'})
    config = await ArbitraryJobConfiguration.from_template_and_values(
        base_job_template=template, values=overrides
    )
    assert config.model_dump() == expected


async def test_job_configuration_from_template_and_overrides_with_nested_variables() -> None:
    template: Dict[str, Any] = {
        'job_configuration': {'config': {'var1': '{{ var1 }}', 'var2': '{{ var2 }}'}},
        'variables': {'properties': {
            'var1': {'type': 'string', 'title': 'Var1'},
            'var2': {'type': 'integer', 'title': 'Var2', 'default': 42}
        }},
        'required': ['var1']
    }

    class ArbitraryJobConfiguration(BaseJobConfiguration):
        config: Dict[str, Any] = Field(json_schema_extra={'template': {'var1': '{{ var1 }}', 'var2': '{{ var2 }}'}}, default_factory=dict)
    config_obj = await ArbitraryJobConfiguration.from_template_and_values(base_job_template=template, values={'var1': 'woof!'})
    assert config_obj.model_dump() == {'command': None, 'env': {}, 'labels': {}, 'name': None, 'config': {'var1': 'woof!', 'var2': 42}}


async def test_job_configuration_from_template_and_overrides_with_hard_coded_primitives() -> None:
    template: Dict[str, Any] = {
        'job_configuration': {'config': {'var1': 1, 'var2': 1.1, 'var3': True}},
        'variables': {}
    }

    class ArbitraryJobConfiguration(BaseJobConfiguration):
        config: Dict[str, Any] = Field(json_schema_extra={'template': {'var1': 1, 'var2': 1.1, 'var3': True}})
    config_obj = await ArbitraryJobConfiguration.from_template_and_values(base_job_template=template, values={})
    assert config_obj.model_dump() == {'command': None, 'env': {}, 'labels': {}, 'name': None, 'config': {'var1': 1, 'var2': 1.1, 'var3': True}}


async def test_job_configuration_from_template_overrides_with_block() -> None:
    class ArbitraryBlock(Block):
        pass
    template: Dict[str, Any] = {
        'job_configuration': {'var1': '{{ var1 }}', 'arbitrary_block': '{{ arbitrary_block }}'},
        'variables': {
            'properties': {
                'var1': {'type': 'string'},
                'arbitrary_block': {}
            },
            'definitions': {
                'ArbitraryBlock': {
                    'title': 'ArbitraryBlock',
                    'type': 'object',
                    'properties': {
                        'a': {'title': 'A', 'type': 'number'},
                        'b': {'title': 'B', 'type': 'string'}
                    },
                    'required': ['a', 'b'],
                    'block_type_slug': 'arbitrary_block',
                    'secret_fields': [],
                    'block_schema_references': {}
                }
            },
            'required': ['var1', 'arbitrary_block']
        }
    }
    class ArbitraryJobConfiguration(BaseJobConfiguration):
        pass
    block_id: uuid.UUID = await ArbitraryBlock(a=1, b='hello').save(name='arbitrary-block')
    config_obj = await ArbitraryJobConfiguration.from_template_and_values(
        base_job_template=template, values={'var1': 'woof!', 'arbitrary_block': {'$ref': {'block_document_id': block_id}}}
    )
    assert config_obj.model_dump() == {
        'command': None,
        'env': {},
        'labels': {},
        'name': None,
        'var1': 'woof!',
        'arbitrary_block': {'a': 1, 'b': 'hello', 'block_type_slug': 'arbitraryblock'}
    }
    config_obj = await ArbitraryJobConfiguration.from_template_and_values(
        base_job_template=template, values={'var1': 'woof!', 'arbitrary_block': '{{ prefect.blocks.arbitraryblock.arbitrary-block }}'}
    )
    assert config_obj.model_dump() == {
        'command': None,
        'env': {},
        'labels': {},
        'name': None,
        'var1': 'woof!',
        'arbitrary_block': {'a': 1, 'b': 'hello', 'block_type_slug': 'arbitraryblock'}
    }


async def test_job_configuration_from_template_coerces_work_pool_values() -> None:
    class ArbitraryJobConfiguration(BaseJobConfiguration):
        pass
    test_work_pool_base_job_config: Dict[str, Any] = {
        'job_configuration': {'var1': 'hello', 'env': {'MY_ENV_VAR': 42, 'OTHER_ENV_VAR': None}},
        'variables': {'properties': {'var1': {'type': 'string'}, 'env': {'type': 'object'}}}
    }
    config_obj = await ArbitraryJobConfiguration.from_template_and_values(base_job_template=test_work_pool_base_job_config, values={})
    assert config_obj.model_dump() == {'command': None, 'env': {'MY_ENV_VAR': '42', 'OTHER_ENV_VAR': None}, 'labels': {}, 'name': None, 'var1': 'hello'}
    assert isinstance(config_obj, ArbitraryJobConfiguration)


@pytest.mark.usefixtures('variables')
async def test_job_configuration_from_template_overrides_with_remote_variables() -> None:
    template: Dict[str, Any] = {
        'job_configuration': {'var1': '{{ var1 }}', 'env': '{{ env }}'},
        'variables': {'properties': {'var1': {'type': 'string'}, 'env': {'type': 'object'}}}
    }

    class ArbitraryJobConfiguration(BaseJobConfiguration):
        env: Dict[str, Any] = Field(default_factory=dict)
    config_obj = await ArbitraryJobConfiguration.from_template_and_values(
        base_job_template=template,
        values={'var1': '{{  prefect.variables.test_variable_1 }}', 'env': {'MY_ENV_VAR': '{{  prefect.variables.test_variable_2 }}'}}
    )
    assert config_obj.model_dump() == {
        'command': None, 'env': {'MY_ENV_VAR': 'test_value_2'}, 'labels': {}, 'name': None, 'var1': 'test_value_1'
    }


@pytest.mark.usefixtures('variables')
async def test_job_configuration_from_template_overrides_with_remote_variables_hardcodes() -> None:
    template: Dict[str, Any] = {
        'job_configuration': {'var1': '{{ prefect.variables.test_variable_1 }}', 'env': {'MY_ENV_VAR': '{{ prefect.variables.test_variable_2 }}'}},
        'variables': {'properties': {}}
    }

    class ArbitraryJobConfiguration(BaseJobConfiguration):
        pass
    config_obj = await ArbitraryJobConfiguration.from_template_and_values(base_job_template=template, values={})
    assert config_obj.model_dump() == {
        'command': None, 'env': {'MY_ENV_VAR': 'test_value_2'}, 'labels': {}, 'name': None, 'var1': 'test_value_1'
    }


async def test_job_configuration_from_template_and_overrides_with_variables_in_a_list() -> None:
    template: Dict[str, Any] = {
        'job_configuration': {'config': ['{{ var1 }}', '{{ var2 }}']},
        'variables': {
            'properties': {
                'var1': {'type': 'string', 'title': 'Var1'},
                'var2': {'type': 'integer', 'title': 'Var2', 'default': 42}
            }
        },
        'required': ['var1']
    }

    class ArbitraryJobConfiguration(BaseJobConfiguration):
        config: Any = Field(json_schema_extra={'template': ['{{ var1 }}', '{{ var2 }}']})
    config_obj = await ArbitraryJobConfiguration.from_template_and_values(base_job_template=template, values={'var1': 'woof!'})
    assert config_obj.model_dump() == {'command': None, 'env': {}, 'labels': {}, 'name': None, 'config': ['woof!', 42]}


@pytest.mark.parametrize('falsey_value', [None, ''])
async def test_base_job_configuration_converts_falsey_values_to_none(falsey_value: Optional[Any]) -> None:
    """Test that valid falsey values are converted to None for `command`"""
    template: Dict[str, Any] = {
        'job_configuration': {'command': '{{ command }}'},
        'variables': {
            'properties': {
                'command': {'type': 'string', 'title': 'Command'}
            },
            'required': []
        }
    }
    config = await BaseJobConfiguration.from_template_and_values(base_job_template=template, values={'command': falsey_value})
    assert config.command is None


@pytest.mark.parametrize('field_template_value,expected_final_template', [
    ('{{ var1 }}', {'command': '{{ command }}', 'env': '{{ env }}', 'labels': '{{ labels }}', 'name': '{{ name }}', 'var1': '{{ var1 }}', 'var2': '{{ var2 }}'}),
    (None, {'command': '{{ command }}', 'env': '{{ env }}', 'labels': '{{ labels }}', 'name': '{{ name }}', 'var1': '{{ var1 }}', 'var2': '{{ var2 }}'}),
    ('{{ dog }}', {'command': '{{ command }}', 'env': '{{ env }}', 'labels': '{{ labels }}', 'name': '{{ name }}', 'var1': '{{ dog }}', 'var2': '{{ var2 }}'})
])
def test_job_configuration_produces_correct_json_template(field_template_value: Any, expected_final_template: Dict[str, Any]) -> None:
    class ArbitraryJobConfiguration(BaseJobConfiguration):
        var1: Optional[str] = Field(json_schema_extra={'template': field_template_value})
        var2: Optional[str] = Field(json_schema_extra={'template': '{{ var2 }}'})
    template = ArbitraryJobConfiguration.json_template()
    assert template == expected_final_template


class TestWorkerProperties:
    def test_defaults(self) -> None:
        class WorkerImplNoCustomization(BaseWorker):
            type: str = 'test-no-customization'
            async def run(self) -> None:
                pass
            async def verify_submitted_deployment(self, deployment: Any) -> None:
                pass
        assert WorkerImplNoCustomization.get_logo_url() == ''
        assert WorkerImplNoCustomization.get_documentation_url() == ''
        assert WorkerImplNoCustomization.get_description() == ''
        assert WorkerImplNoCustomization.get_default_base_job_template() == {
            'job_configuration': {
                'command': '{{ command }}',
                'env': '{{ env }}',
                'labels': '{{ labels }}',
                'name': '{{ name }}'
            },
            'variables': {
                'properties': {
                    'command': {
                        'anyOf': [{'type': 'string'}, {'type': 'null'}],
                        'title': 'Command',
                        'default': None,
                        'description': 'The command to use when starting a flow run. In most cases, this should be left blank and the command will be automatically generated by the worker.'
                    },
                    'env': {
                        'title': 'Environment Variables',
                        'type': 'object',
                        'additionalProperties': {'anyOf': [{'type': 'string'}, {'type': 'null'}]},
                        'description': 'Environment variables to set when starting a flow run.'
                    },
                    'labels': {
                        'title': 'Labels',
                        'type': 'object',
                        'additionalProperties': {'type': 'string'},
                        'description': 'Labels applied to infrastructure created by the worker using this job configuration.'
                    },
                    'name': {
                        'anyOf': [{'type': 'string'}, {'type': 'null'}],
                        'default': None,
                        'title': 'Name',
                        'description': 'Name given to infrastructure created by the worker using this job configuration.'
                    }
                },
                'type': 'object'
            }
        }

    def test_custom_logo_url(self) -> None:
        class WorkerImplWithLogoUrl(BaseWorker):
            type: str = 'test-with-logo-url'
            job_configuration: Type[BaseJobConfiguration] = BaseJobConfiguration
            _logo_url: str = 'https://example.com/logo.png'
            async def run(self) -> None:
                pass
            async def verify_submitted_deployment(self, deployment: Any) -> None:
                pass
        assert WorkerImplWithLogoUrl.get_logo_url() == 'https://example.com/logo.png'

    def test_custom_documentation_url(self) -> None:
        class WorkerImplWithDocumentationUrl(BaseWorker):
            type: str = 'test-with-documentation-url'
            job_configuration: Type[BaseJobConfiguration] = BaseJobConfiguration
            _documentation_url: str = 'https://example.com/docs'
            async def run(self) -> None:
                pass
            async def verify_submitted_deployment(self, deployment: Any) -> None:
                pass
        assert WorkerImplWithDocumentationUrl.get_documentation_url() == 'https://example.com/docs'

    def test_custom_description(self) -> None:
        class WorkerImplWithDescription(BaseWorker):
            type: str = 'test-with-description'
            job_configuration: Type[BaseJobConfiguration] = BaseJobConfiguration
            _description: str = 'Custom Worker Description'
            async def run(self) -> None:
                pass
            async def verify_submitted_deployment(self, deployment: Any) -> None:
                pass
        assert WorkerImplWithDescription.get_description() == 'Custom Worker Description'

    def test_custom_base_job_configuration(self) -> None:
        class CustomBaseJobConfiguration(BaseJobConfiguration):
            var1: Optional[str] = Field(json_schema_extra={'template': '{{ var1 }}'})
            var2: Optional[int] = Field(json_schema_extra={'template': '{{ var2 }}'})
        class CustomBaseVariables(BaseVariables):
            var1: Any
            var2: int = 1
        class WorkerImplWithCustomBaseJobConfiguration(BaseWorker):
            type: str = 'test-with-base-job-configuration'
            job_configuration: Type[BaseJobConfiguration] = CustomBaseJobConfiguration
            job_configuration_variables: Type[BaseVariables] = CustomBaseVariables
            async def run(self) -> None:
                pass
            async def verify_submitted_deployment(self, deployment: Any) -> None:
                pass
        assert WorkerImplWithCustomBaseJobConfiguration.get_default_base_job_template() == {
            'job_configuration': {
                'command': '{{ command }}',
                'env': '{{ env }}',
                'labels': '{{ labels }}',
                'name': '{{ name }}',
                'var1': '{{ var1 }}',
                'var2': '{{ var2 }}'
            },
            'variables': {
                'properties': {
                    'command': {
                        'title': 'Command',
                        'anyOf': [{'type': 'string'}, {'type': 'null'}],
                        'default': None,
                        'description': 'The command to use when starting a flow run. In most cases, this should be left blank and the command will be automatically generated by the worker.'
                    },
                    'env': {
                        'title': 'Environment Variables',
                        'type': 'object',
                        'additionalProperties': {'anyOf': [{'type': 'string'}, {'type': 'null'}]},
                        'description': 'Environment variables to set when starting a flow run.'
                    },
                    'labels': {
                        'title': 'Labels',
                        'type': 'object',
                        'additionalProperties': {'type': 'string'},
                        'description': 'Labels applied to infrastructure created by a worker.'
                    },
                    'name': {
                        'title': 'Name',
                        'anyOf': [{'type': 'string'}, {'type': 'null'}],
                        'default': None,
                        'description': 'Name given to infrastructure created by a worker.'
                    },
                    'var1': {'title': 'Var1', 'type': 'string'},
                    'var2': {'title': 'Var2', 'type': 'integer', 'default': 1}
                },
                'required': ['var1'],
                'type': 'object'
            }
        }


class TestPrepareForFlowRun:
    @pytest.fixture
    def job_config(self) -> BaseJobConfiguration:
        return BaseJobConfiguration(env={'MY_VAR': 'foo'}, labels={'my-label': 'foo'}, name='my-job-name')

    @pytest.fixture
    def flow_run(self) -> FlowRun:
        return FlowRun(name='my-flow-run-name', flow_id=uuid.uuid4())

    @pytest.fixture
    def flow(self) -> Flow:
        return Flow(name='my-flow-name')

    @pytest.fixture
    def deployment(self, flow: Flow) -> DeploymentResponse:
        return DeploymentResponse(name='my-deployment-name', flow_id=flow.id)

    def test_prepare_for_flow_run_without_deployment_and_flow(self, job_config: BaseJobConfiguration, flow_run: FlowRun) -> None:
        job_config.prepare_for_flow_run(flow_run)
        assert job_config.env == {**get_current_settings().to_environment_variables(exclude_unset=True), 'MY_VAR': 'foo', 'PREFECT__FLOW_RUN_ID': str(flow_run.id)}
        assert job_config.labels == {'my-label': 'foo', 'prefect.io/flow-run-id': str(flow_run.id), 'prefect.io/flow-run-name': flow_run.name, 'prefect.io/version': prefect.__version__}
        assert job_config.name == 'my-job-name'
        assert job_config.command == 'prefect flow-run execute'

    def test_prepare_for_flow_run(self, job_config: BaseJobConfiguration, flow_run: FlowRun) -> None:
        job_config.prepare_for_flow_run(flow_run)
        assert job_config.env == {**get_current_settings().to_environment_variables(exclude_unset=True), 'MY_VAR': 'foo', 'PREFECT__FLOW_RUN_ID': str(flow_run.id)}
        assert job_config.labels == {'my-label': 'foo', 'prefect.io/flow-run-id': str(flow_run.id), 'prefect.io/flow-run-name': flow_run.name, 'prefect.io/version': prefect.__version__}
        assert job_config.name == 'my-job-name'
        assert job_config.command == 'prefect flow-run execute'

    def test_prepare_for_flow_run_with_deployment_and_flow(self, job_config: BaseJobConfiguration, flow_run: FlowRun, deployment: DeploymentResponse, flow: Flow) -> None:
        job_config.prepare_for_flow_run(flow_run, deployment=deployment, flow=flow)
        assert job_config.env == {**get_current_settings().to_environment_variables(exclude_unset=True), 'MY_VAR': 'foo', 'PREFECT__FLOW_RUN_ID': str(flow_run.id)}
        assert job_config.labels == {
            'my-label': 'foo',
            'prefect.io/flow-run-id': str(flow_run.id),
            'prefect.io/flow-run-name': flow_run.name,
            'prefect.io/version': prefect.__version__,
            'prefect.io/deployment-id': str(deployment.id),
            'prefect.io/deployment-name': deployment.name,
            'prefect.io/flow-id': str(flow.id),
            'prefect.io/flow-name': flow.name
        }
        assert job_config.name == 'my-job-name'
        assert job_config.command == 'prefect flow-run execute'


async def test_get_flow_run_logger_without_worker_id_set(prefect_client: PrefectClient, worker_deployment_wq1: Any, work_pool: WorkPool) -> None:
    flow_run = await prefect_client.create_flow_run_from_deployment(worker_deployment_wq1.id)
    async with WorkerTestImpl(name='test', work_pool_name=work_pool.name, create_pool_if_not_found=False) as worker:
        await worker.sync_with_backend()
        assert worker.backend_id is None
        logger = worker.get_flow_run_logger(flow_run)
        assert logger.name == 'prefect.flow_runs.worker'
        assert logger.extra == {
            'flow_run_name': flow_run.name,
            'flow_run_id': str(flow_run.id),
            'flow_name': '<unknown>',
            'worker_name': 'test',
            'work_pool_name': work_pool.name,
            'work_pool_id': str(work_pool.id)
        }


async def test_get_flow_run_logger_with_worker_id_set(prefect_client: PrefectClient, worker_deployment_wq1: Any, work_pool: WorkPool) -> None:
    flow_run = await prefect_client.create_flow_run_from_deployment(worker_deployment_wq1.id)
    async with WorkerTestImpl(name='test', work_pool_name=work_pool.name, create_pool_if_not_found=False) as worker:
        await worker.sync_with_backend()
        worker_id: uuid.UUID = uuid.uuid4()
        worker.backend_id = worker_id
        logger = worker.get_flow_run_logger(flow_run)
        assert logger.name == 'prefect.flow_runs.worker'
        assert logger.extra == {
            'flow_run_name': flow_run.name,
            'flow_run_id': str(flow_run.id),
            'flow_name': '<unknown>',
            'worker_name': 'test',
            'work_pool_name': work_pool.name,
            'work_pool_id': str(work_pool.id),
            'worker_id': str(worker_id)
        }


class TestInfrastructureIntegration:
    async def test_worker_crashes_flow_if_infrastructure_submission_fails(
        self, prefect_client: PrefectClient, worker_deployment_infra_wq1: Any, work_pool: WorkPool, monkeypatch: Any
    ) -> None:
        flow_run = await prefect_client.create_flow_run_from_deployment(worker_deployment_infra_wq1.id, state=Scheduled(scheduled_time=pendulum.now('utc')))
        await prefect_client.read_flow(worker_deployment_infra_wq1.flow_id)

        def raise_value_error() -> None:
            raise ValueError('Hello!')
        mock_run = MagicMock()
        mock_run.run = raise_value_error
        async with WorkerTestImpl(work_pool_name=work_pool.name) as worker:
            worker._work_pool = work_pool
            monkeypatch.setattr(worker, 'run', mock_run)
            monkeypatch.setattr(worker, 'run', mock_run)
            await worker.get_and_submit_flow_runs()
        state = (await prefect_client.read_flow_run(flow_run.id)).state
        assert state.is_crashed()
        with pytest.raises(CrashedRun, match='Flow run could not be submitted to infrastructure'):
            await state.result()


async def test_worker_set_last_polled_time(work_pool: WorkPool) -> None:
    now = pendulum.now('utc')
    if version.parse(pendulum.__version__) >= version.parse('3.0'):
        with pendulum.travel_to(now, freeze=True):
            async with WorkerTestImpl(work_pool_name=work_pool.name) as worker:
                assert worker._last_polled_time == now
                now2 = now.add(seconds=49)
                with pendulum.travel_to(now2, freeze=True):
                    await worker.get_and_submit_flow_runs()
                    assert worker._last_polled_time == now2
                now3 = pendulum.datetime(2021, 1, 1, 0, 0, 0, tz='utc')
                with pendulum.travel_to(now3, freeze=True):
                    await worker.get_and_submit_flow_runs()
                    assert worker._last_polled_time == now3
    else:
        pendulum.set_test_now(now)
        async with WorkerTestImpl(work_pool_name=work_pool.name) as worker:
            assert worker._last_polled_time == now
            now2 = now.add(seconds=49)
            pendulum.set_test_now(now2)
            await worker.get_and_submit_flow_runs()
            assert worker._last_polled_time == now2
            now3 = pendulum.datetime(2021, 1, 1, 0, 0, 0, tz='utc')
            pendulum.set_test_now(now3)
            await worker.get_and_submit_flow_runs()
            assert worker._last_polled_time == now3
            pendulum.set_test_now()


async def test_worker_last_polled_health_check(work_pool: WorkPool) -> None:
    now = pendulum.now('utc')
    if version.parse(pendulum.__version__) >= version.parse('3.0'):
        pendulum.travel_to(now, freeze=True)
        async with WorkerTestImpl(work_pool_name=work_pool.name) as worker:
            resp = worker.is_worker_still_polling(query_interval_seconds=10)
            assert resp is True
            with pendulum.travel(seconds=299):
                resp = worker.is_worker_still_polling(query_interval_seconds=10)
                assert resp is True
            with pendulum.travel(seconds=301):
                resp = worker.is_worker_still_polling(query_interval_seconds=10)
                assert resp is False
            with pendulum.travel(minutes=30):
                resp = worker.is_worker_still_polling(query_interval_seconds=60)
                assert resp is True
            with pendulum.travel(minutes=30, seconds=1):
                resp = worker.is_worker_still_polling(query_interval_seconds=60)
                assert resp is False
    else:
        pendulum.set_test_now(now)
        async with WorkerTestImpl(work_pool_name=work_pool.name) as worker:
            resp = worker.is_worker_still_polling(query_interval_seconds=10)
            assert resp is True
            pendulum.set_test_now(now.add(seconds=299))
            resp = worker.is_worker_still_polling(query_interval_seconds=10)
            assert resp is True
            pendulum.set_test_now(now.add(seconds=301))
            resp = worker.is_worker_still_polling(query_interval_seconds=10)
            assert resp is False
            pendulum.set_test_now(now.add(minutes=30))
            resp = worker.is_worker_still_polling(query_interval_seconds=60)
            assert resp is True
            pendulum.set_test_now(now.add(minutes=30, seconds=1))
            resp = worker.is_worker_still_polling(query_interval_seconds=60)
            assert resp is False
            pendulum.set_test_now()


class TestBaseWorkerStart:
    async def test_start_syncs_with_the_server(self, work_pool: WorkPool) -> None:
        worker = WorkerTestImpl(work_pool_name=work_pool.name)
        assert worker._work_pool is None
        await worker.start(run_once=True)
        assert worker._work_pool is not None
        assert worker._work_pool.base_job_template == work_pool.base_job_template

    async def test_start_executes_flow_runs(self, prefect_client: PrefectClient, worker_deployment_wq1: Any, work_pool: WorkPool) -> None:
        @flow
        def test_flow() -> None:
            pass

        def create_run_with_deployment(state: Any) -> Any:
            return prefect_client.create_flow_run_from_deployment(worker_deployment_wq1.id, state=state)
        flow_run = await prefect_client.create_flow_run_from_deployment(worker_deployment_wq1.id, state=Scheduled(scheduled_time=pendulum.now('utc').subtract(days=1)))
        worker = WorkerTestImpl(work_pool_name=work_pool.name)
        worker.run = AsyncMock()
        await worker.start(run_once=True)
        worker.run.assert_awaited_once()
        assert worker.run.call_args[1]['flow_run'].id == flow_run.id


@pytest.mark.parametrize('work_pool_env, deployment_env, flow_run_env, expected_env', [
    ({}, {'test-var': 'foo'}, {'another-var': 'boo'}, {'test-var': 'foo', 'another-var': 'boo'}),
    ({'A': '1', 'B': '2'}, {'A': '1', 'B': '3'}, {}, {'A': '1', 'B': '3'}),
    ({'A': '1', 'B': '2'}, {'C': '3', 'D': '4'}, {}, {'A': '1', 'B': '2', 'C': '3', 'D': '4'}),
    ({'A': '1', 'B': '2'}, {'C': '42'}, {'C': '3', 'D': '4'}, {'A': '1', 'B': '2', 'C': '3', 'D': '4'}),
    ({'A': '1', 'B': '2'}, {'B': ''}, {}, {'A': '1', 'B': ''})
], ids=['flow_run_into_deployment', 'deployment_into_work_pool_overlap', 'deployment_into_work_pool_no_overlap', 'flow_run_into_work_pool', 'try_overwrite_with_empty_str'])
@pytest.mark.parametrize('use_variable_defaults', [True, False], ids=['with_defaults', 'without_defaults'])
async def test_env_merge_logic_is_deep(
    prefect_client: PrefectClient,
    session: Any,
    flow: Flow,
    work_pool: WorkPool,
    work_pool_env: Dict[str, Any],
    deployment_env: Dict[str, Any],
    flow_run_env: Dict[str, Any],
    expected_env: Dict[str, Any],
    use_variable_defaults: bool
) -> None:
    if work_pool_env:
        base_job_template: Dict[str, Any] = (
            {'job_configuration': {'env': '{{ env }}'}, 'variables': {'properties': {'env': {'type': 'object', 'default': work_pool_env}}}}
            if use_variable_defaults
            else {'job_configuration': {'env': work_pool_env}, 'variables': {'properties': {'env': {'type': 'object'}}}}
        )
        await models.workers.update_work_pool(
            session=session, work_pool_id=work_pool.id,
            work_pool=ServerWorkPoolUpdate(base_job_template=base_job_template, description='test', is_paused=False, concurrency_limit=None)
        )
        await session.commit()
    deployment = await models.deployments.create_deployment(
        session=session,
        deployment=Deployment(name='env-testing', tags=['test'], flow_id=flow.id, path='./subdir', entrypoint='/file.py:flow', parameter_openapi_schema={}, job_variables={'env': deployment_env}, work_queue_id=work_pool.default_queue_id)
    )
    await session.commit()
    flow_run = await prefect_client.create_flow_run_from_deployment(deployment.id, state=Pending(), job_variables={'env': flow_run_env})
    async with WorkerTestImpl(name='test', work_pool_name=work_pool.name if work_pool_env else 'test-work-pool') as worker:
        await worker.sync_with_backend()
        config = await worker._get_configuration(flow_run, schemas.responses.DeploymentResponse.model_validate(deployment))
    for key, value in expected_env.items():
        assert config.env[key] == value


class TestBaseWorkerHeartbeat:
    async def test_worker_heartbeat_sends_integrations(self, work_pool: WorkPool, hosted_api_server: Any) -> None:
        async with WorkerTestImpl(work_pool_name=work_pool.name) as worker:
            await worker.start(run_once=True)
            with mock.patch('prefect.workers.base.load_prefect_collections') as mock_load_prefect_collections, \
                 mock.patch('prefect.client.orchestration._work_pools.client.WorkPoolAsyncClient.send_worker_heartbeat') as mock_send_worker_heartbeat_post, \
                 mock.patch('prefect.workers.base.distributions') as mock_distributions:
                mock_load_prefect_collections.return_value = {'prefect_aws': '1.0.0'}
                mock_distributions.return_value = [mock.MagicMock(metadata={'Name': 'prefect-aws'}, version='1.0.0')]
                async with get_client() as client:
                    worker._client = client
                    worker._client.server_type = ServerType.CLOUD
                    await worker.sync_with_backend()
                mock_send_worker_heartbeat_post.assert_called_once_with(
                    work_pool_name=work_pool.name,
                    worker_name=worker.name,
                    heartbeat_interval_seconds=30.0,
                    get_worker_id=True,
                    worker_metadata=WorkerMetadata(integrations=[Integration(name='prefect-aws', version='1.0.0')])
                )
            assert worker._worker_metadata_sent

    async def test_custom_worker_can_send_arbitrary_metadata(self, work_pool: WorkPool, hosted_api_server: Any) -> None:
        class CustomWorker(BaseWorker):
            type: str = 'test-custom-metadata'
            job_configuration: Type[BaseJobConfiguration] = BaseJobConfiguration
            async def run(self) -> None:
                pass
            async def _worker_metadata(self) -> WorkerMetadata:
                return WorkerMetadata(**{'integrations': [{'name': 'prefect-aws', 'version': '1.0.0'}], 'custom_field': 'heya'})
        async with CustomWorker(work_pool_name=work_pool.name) as worker:
            await worker.start(run_once=True)
            with mock.patch('prefect.workers.base.load_prefect_collections') as mock_load_prefect_collections, \
                 mock.patch('prefect.client.orchestration._work_pools.client.WorkPoolAsyncClient.send_worker_heartbeat') as mock_send_worker_heartbeat_post, \
                 mock.patch('prefect.workers.base.distributions') as mock_distributions:
                mock_load_prefect_collections.return_value = {'prefect_aws': '1.0.0'}
                mock_distributions.return_value = [mock.MagicMock(metadata={'Name': 'prefect-aws'}, version='1.0.0')]
                async with get_client() as client:
                    worker._client = client
                    worker._client.server_type = ServerType.CLOUD
                    await worker.sync_with_backend()
                mock_send_worker_heartbeat_post.assert_called_once_with(
                    work_pool_name=work_pool.name,
                    worker_name=worker.name,
                    heartbeat_interval_seconds=30.0,
                    get_worker_id=True,
                    worker_metadata=WorkerMetadata(integrations=[Integration(name='prefect-aws', version='1.0.0')], custom_field='heya')
                )
            assert worker._worker_metadata_sent


async def test_worker_gives_labels_to_flow_runs_when_using_cloud_api(prefect_client: PrefectClient, worker_deployment_wq1: Any, work_pool: WorkPool) -> None:
    def create_run_with_deployment(state: Any) -> Any:
        return prefect_client.create_flow_run_from_deployment(worker_deployment_wq1.id, state=state)
    flow_run = await create_run_with_deployment(Scheduled(scheduled_time=pendulum.now('utc').subtract(days=1)))
    async with WorkerTestImpl(work_pool_name=work_pool.name) as worker:
        assert worker._client is not None
        worker._client.server_type = ServerType.CLOUD
        worker._work_pool = work_pool
        worker.run = AsyncMock()
        await worker.get_and_submit_flow_runs()
    flow_run = await prefect_client.read_flow_run(flow_run.id)
    expected_labels: Dict[str, str] = {
        'prefect.worker.name': worker.name,
        'prefect.worker.type': worker.type,
        'prefect.work-pool.name': work_pool.name,
        'prefect.work-pool.id': str(work_pool.id)
    }
    for key, value in expected_labels.items():
        assert flow_run.labels[key] == value


async def test_worker_removes_flow_run_from_submitting_when_not_ready(prefect_client: PrefectClient, worker_deployment_wq1: Any, work_pool: WorkPool) -> None:
    """
    Regression test for https://github.com/PrefectHQ/prefect/issues/16027
    """
    flow_run = await prefect_client.create_flow_run_from_deployment(worker_deployment_wq1.id, state=Pending())
    async with WorkerTestImpl(work_pool_name=work_pool.name) as worker:
        worker._propose_pending_state = AsyncMock(return_value=False)
        await worker.get_and_submit_flow_runs()
        assert flow_run.id not in worker._submitting_flow_run_ids
